import logging 
import os 
import sys 


DEBUG = False 

# Action Vocabulary
VOCABULARY = {}
VOCABULARY['cross_acts'] = {'close', 'near', 'far', 'approach', 'leave', 'cross'}
VOCABULARY['single_acts'] = {'start', 'end', 'move', 'stop', 'use_phone', 'carry', 
                            'use_computer', 'give', 'talk', 'sit', 'with_bike', 'with_bag'}

class Act:
    def __init__(self, act_name='', class1='', tube1='', class2='', tube2='', frame_id=0):
        self.act_name = act_name
        self.class1 = class1
        self.tube1 = tube1
        self.class2 = class2
        self.tube2 = tube2
        
        # frame_id is the starting frame number of the act
        self.frame_id = frame_id
        self.matched = False 
        self.valid = False

    def to_meta(self):
        res = {
            "id": self.class1 + '-' + str(self.tube1),
            "label": self.act_name,
            "act_fid": self.frame_id,
        }
        if self.tube2:
            res["id2"] = self.class2 + '-' + str(self.tube2)

        return res

    def to_log(self):
        res = self.class1 + '-' + str(self.tube1) + ':' + self.act_name
        if self.tube2:
            res += ':' + self.class2 + '-' + str(self.tube2)

        return res 

    def log(self, s):
        logging.debug('[Act] %s' % s )


ACT_STARTER = '>>'
def load_graph_from_file(fname):
    assert os.path.exists(fname), 'act config %s not exists!' % fname
    lines = open(fname, 'r').readlines()
    act_def = []
    act_name = ''
    graphs = []
    for line in lines:
        data = line.strip()
        if not data or '#' in data:
            continue
        if ACT_STARTER in data:
            if act_name and act_def:
                def_str = '\n'.join(act_def)
                g = ActGraph(act_name, def_str)
                act_def = []
                if g.valid:
                    graphs.append(g)
            act_name = data.split(ACT_STARTER)[1].strip()
        else:
            act_def.append(data)

    if act_name and act_def:
        def_str = '\n'.join(act_def)
        g = ActGraph(act_name, def_str)
        if g.valid:
            graphs.append(g)

    return graphs


class ActGraph:
    def __init__(self, name, definition):
        '''
        Args:
        - name: the action name
        - defition: the string that use syntax to define the graph
        '''
        self.name = name
        self.acts = []
        self.subject_dict = {}

        self.acts_pointer = 0
        self.completed = False 
        self.valid = True 

        name_def = []
        logic_def = []
        defs = definition.split('\n')
        for d in defs:
            data = d.strip()
            if not data or '#' in data: 
                continue
            if '=' in data:
                name_def.append(data)
            else:
                logic_def.append(data)

        self.parse_subjects(name_def)
        self.parse_logic(logic_def)
        self.log('Subject: %s' % str(self.subject_dict))

    def parse_subjects(self, name_def):
        for line in name_def:
            pairs = line.strip().split('=')
            if len(pairs) != 2: 
                continue
            k, v = pairs[0].strip(), pairs[1].strip()
            if not (k and v):
                continue 
            
            if k in self.subject_dict:
                self.error('dup subject! %s' % str(name_def))
                self.valid = False 
                break 

            self.subject_dict[k] = {
                'type': v,
                'id': ''
            }

        self.valid = False if not len(self.subject_dict) else True 

    def parse_logic(self, logic_def):
        for line in logic_def:
            and_states = line.split(' or ')
            then_tier = []
            for statement in and_states:
                then_tier.append(self.parse_and(statement.strip()))

            if then_tier:
                self.acts.append(then_tier)
            else:
                self.valid = False 

    def parse_and(self, statement):
        ''' Extract each atomic act in the statement string
        '''
        res = []
        cache = []

        def _gen_act_from_cache(cache, res):
            if not cache:
                return 
            act = self.build_act_from_str(''.join(cache))
            if act.valid: 
                res.append(act)

        for c in statement:
            if c == '(':
                cache = []
            elif c == ')':
                _gen_act_from_cache(cache, res)
                cache = []
            else:
                cache.append(c)

        _gen_act_from_cache(cache, res)

        if not res:
            self.valid = False 
        return res

    def match(self, act, read_only=False):
        """
        Match an act to the graph, update the acts in the graph,
        and the acts_pointer. If set read_only, we will only look
        for matching actions, but not update the graph itself.

        Return True if a matched act in graph is found 
        """
        matched = False 
        matched_num = 0 
        for i, a in enumerate(self.acts[self.acts_pointer][0]):
            if a.matched:
                matched_num += 1
                continue 

            if a.act_name != act.act_name or a.class1 != act.class1:
                continue 
            a_tube1 = self.subject_dict[a.tube1]['id']
            if a_tube1 and a_tube1 != act.tube1:
                continue

            if a.class2:
                a_tube2 = self.subject_dict[a.tube2]['id']
                if a.class2 != act.class2:
                    continue 
                if a_tube2 and a_tube2 != act.tube2:
                    continue 

            conflict_assignment = False 
            for name, data in self.subject_dict.items():
                if data['id'] == act.tube1 and name != a.tube1 or \
                    a.class2 and data['id'] == act.tube2 and name != a.tube2:
                    conflict_assignment = True 
                    break   

            if conflict_assignment:
                continue 
                
            matched = True 
            if read_only:
                break 

            a.matched = True 
            if DEBUG:
                self.log('%s (%s,%s) s%d/%d:m%d/%d - %s' % (
                        a.to_log(), 
                        act.tube1, 
                        act.tube2, 
                        self.acts_pointer, 
                        len(self.acts), 
                        matched_num,
                        len(self.acts[self.acts_pointer][0]),
                        str(self.subject_dict)
                    )
                )
            self.subject_dict[a.tube1]['id'] = act.tube1
            if a.class2:
                self.subject_dict[a.tube2]['id'] = act.tube2

            # Last act got matched 
            if matched_num == len(self.acts[self.acts_pointer][0]) - 1:
                self.acts_pointer += 1
            break 

        if self.acts_pointer >= len(self.acts):
            self.completed = True 
        return matched

    def build_act_from_str(self, line):
        '''
         a rule def string, extract the keywords for the Act
        '''

        data_raw = line.split(' ')
        data = [d for d in data_raw if d]
        act = Act()

        if len(data) < 2 or len(data) > 3:
            self.log("cannot build act from: " + line)
            return act

        valid_cnt = 0
        if data[0] in self.subject_dict:
            act.tube1 = data[0]
            act.class1 = self.subject_dict[data[0]]['type']
            valid_cnt += 1

        if data[1] in VOCABULARY['single_acts'] or data[1] in VOCABULARY['cross_acts']:
            act.act_name = data[1]
            valid_cnt += 1

        if len(data) == 3 and act.act_name in VOCABULARY['cross_acts'] and data[2] in self.subject_dict:
            act.tube2 = data[2]
            act.class2 = self.subject_dict[data[2]]['type']
            valid_cnt += 1

        if valid_cnt == len(data):
            act.valid = True

        return act

    def show_graph_definition(self):
        res = ['>> %s' % self.name]
        for and_states in self.acts:
            tmp = []
            for statement in and_states:
                tmp.append([a.to_log() for a in statement])
            res.append(tmp)
        return res

    def show_name(self):
        return self.name + ':' + str(self.subject_dict)

    def to_act(self):
        ''' select the frist two person as the act performer... '''
        sub_list = []
        for _, sub in self.subject_dict.items():
            if sub['type'] != 'person':
                continue
            sub_list.append(sub['id'])
        
        act = Act(act_name=self.name)
        if sub_list:
            act.class1 = 'person'
            act.tube1 = sub_list[0]
        if len(sub_list) > 1:
            act.class2 = 'person'
            act.tube2 = sub_list[1]

        return act

    def log(self, s):
        logging.debug('[Graph-%s] %s' % (self.name, str(s)))


if __name__ == '__main__':
    logging.basicConfig(filename='graph_debug.log',
                        format='%(asctime)s %(message)s',
                        datefmt='%I:%M:%S ',
                        filemode='w',
                        level=logging.DEBUG)
    name = 'hehe'
    definition = '''
        p1 = Person
        p2 = Person
        (p1 approach p2) and (p1 far p2)
        (p1 close p2)
        (p1 leave p2) and (p1 far p2)
    '''
    print(definition)
    print('--------------')
    print('Output:')
    a = ActGraph(name, definition)
    for line in a.show_graph_definition():
        print(line)
    print(a.valid)


    file_name = sys.argv[1]
    gs = load_graph_from_file(file_name)
    for g in gs:
        print('Graph: %s' % str(g.name))