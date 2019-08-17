from collections import deque, defaultdict
from time import time, sleep
from threading import Thread 
from copy import deepcopy
import logging
from server.action_graph import Act, load_graph_from_file


MAX_GRAPH_CACHE_SIZE = 1000
ACT_DEF_PATH = 'config/act_def.txt'

class CompActDetector:
    def __init__(self, in_queue, out_queue, filter_queue):
        self.in_queue = in_queue
        self.out_queue = out_queue
        # this queue includes all the tube ids that we need to run NN on
        self.filter_queue = filter_queue    

        self.default_graphs = load_graph_from_file(ACT_DEF_PATH)
        for g in self.default_graphs:
            self.log('Load graph: %s' % str(g.show_graph_definition()))
        self.active_graphs = deque()
        self.id_actions = defaultdict(list) # cid/tid -> [act involved the cid/tid]

    def run(self):
        while True:
            server_pkt = self.in_queue.read()
            if server_pkt is None:
                sleep(0.01)
                continue

            res = []
            cam_id = server_pkt.cam_id
            for act in server_pkt.actions:
                act.tube1 = "%s|%s" % (cam_id, act.tube1)
                if act.tube2:
                    act.tube2 = "%s|%s" % (cam_id, act.tube2)

                # Activate graphs if an act get matched 
                for graph in self.default_graphs:
                    if graph.match(act, read_only=True):
                        active = deepcopy(graph)
                        active.match(act)
                        if not active.completed:
                            self.active_graphs.append(active)
                        else:
                            res.append(active.to_act())
                            self.log("Completed: %s" % active.show_name())

                # Match act with existing active graphs
                for _ in range(len(self.active_graphs)):
                    active = self.active_graphs.popleft()
                    active.match(act)
                    if not active.completed:
                        self.active_graphs.append(active)
                    else:
                        res.append(active.to_act())
                        self.log("Completed: %s" % active.show_name())

            # Removed active graphs if one stage not fully fullfilled
            for _ in range(len(self.active_graphs)):
                g = self.active_graphs.popleft()
                if g.acts_pointer:
                    self.active_graphs.append(g)

            # Shorten the queue with the capacity threshold 
            while len(self.active_graphs) > MAX_GRAPH_CACHE_SIZE:
                self.active_graphs.popleft()

            self.log('act graph size %d' % len(self.active_graphs))
            server_pkt.actions = res

            # Add newly detected actions to each tube's record 
            for act in res:
                act.tube1 = act.tube1.split('|')[1]
                self.id_actions[cam_id, act.tube1].append(act.act_name)                
                if act.tube2:
                    act.tube2 = act.tube2.split('|')[1]
                    self.id_actions[cam_id, act.tube2].append(act.act_name)

            # Generate reid as an action 
            for cur_tid in server_pkt.reid.keys():
                server_pkt.actions.append(
                    self.get_reid_act(
                        cam_id, 
                        "person", 
                        cur_tid, 
                        server_pkt.reid[cur_tid][0], 
                        server_pkt.reid[cur_tid][1],
                    )
                )

            self.out_queue.write(server_pkt)

    def get_reid_act(self, cam_id, tube_label, tube_id, prev_cid, prev_tid):
        self.log("REID: (%s: %d) -> (%s: %d)" % (prev_cid, prev_tid, cam_id, tube_id))
        return Act(
            "From Cam-%s: %s" % (
                prev_cid, ",".join(self.id_actions[prev_cid, prev_tid][:5])
            ), 
            tube_label, 
            tube_id,
        )

    def log(self, s):
        logging.debug('[CompAct] %s' % s)