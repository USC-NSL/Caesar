import logging 
import sys 
import os 

class Generator:
    def __init__(self, names, img_dim):
        self.names = names
        self.img_wid = img_dim[0]
        self.img_hei = img_dim[1]

        self.python_insert_indicator = '###### AUTO GENERATED #####'
        self.python_insert_template = \
            '        @app.route(\'/%s\')\n        def %s():\n            return Response(read_im(\'%s\'), mimetype=\'multipart/x-mixed-replace; boundary=frame\')\n'
        self.python_src_file = 'web/web_server_template.py'
        self.python_dst_file = 'web/web_server.py'

        self.html_insert_indicator = '<!--##### AUTO GENERATED #####-->'
        self.html_insert_template = '    <img src=\"{{ url_for(\'%s\') }}\" width=\"%d\" height=\"%d\">\n'
        self.html_src_file = 'web/html/index_template.html'
        self.html_dst_file = 'web/html/index.html'


    def generate_code(self, indicator, src, dst, insert_info):
        if not os.path.exists(src):
            self.log('fatal: cannot find source file at %s' % src)
            sys.exit(0)

        lines = open(src, 'r').readlines()
        out = open(dst, 'w')

        for l in lines:
            if not indicator in l:
                out.write(l)
                continue 

            for info in insert_info:
                out.write(info)
                out.write('\n')

        out.close()


    def log(self, s):
        logging.debug('[CodeGenerator] %s' % s)


    def insert_generator(self, template):
        mask = []
        for i in range(len(template) - 1):
            if template[i : i + 2] == '%s':    
                mask.append('s')
            elif template[i : i + 2] == '%d':
                mask.append('d')

        res = []
        for name in self.names:
            temp = []
            first_d = True
            for m in mask:
                if m == 's':
                    temp.append(name)
                elif m == 'd' and first_d:
                    temp.append(self.img_wid)
                    first_d = False
                elif m == 'd' and not first_d:
                    temp.append(self.img_hei)
            res.append(template % tuple(temp))

        return res 


    def run(self):
        self.generate_code(
                    self.python_insert_indicator,
                    self.python_src_file,
                    self.python_dst_file,
                    self.insert_generator(self.python_insert_template)
                )


        self.generate_code(
                    self.html_insert_indicator,
                    self.html_src_file,
                    self.html_dst_file,
                    self.insert_generator(self.html_insert_template)
                )

        self.log('done')


if __name__ == '__main__':
    g = Generator(['lala','hh'], (640, 480))
    g.run()
