import usr
import logging
import tornado.ioloop
import tornado.web

import json
import time
import datetime
import re

drc_args, drf_args, mlm_args = usr.init_args()
drc_args, drc_model, drc_tokenizer, \
drf_args, drf_model, drf_tokenizer, \
mlm_args, mlm_model, mlm_tokenizer = usr.init_models(drc_args, drf_args, mlm_args)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        pass
    
    def post(self):
        body = tornado.escape.json_decode(self.request.body)
        for request in body:
          logging.info('Received Data ' + str(request))

          context = ' '.join([message['text'] for message in request['dialogue_context']])
          fact = request['dialog_fact']
          if fact != '':
            fact = [fact]
          else:
            fact = None
          
          scores = []
          for response in request['response_list']:
              scores.append(usr.get_scores([context], [response], 
                            drc_args, drc_model, drc_tokenizer,
                            drf_args, drf_model, drf_tokenizer,
                            mlm_args, mlm_model, mlm_tokenizer,
                            fact=fact))
          
          request['response_scores'] = scores

          logging.info('Response Scores ' + str(scores))
        self.write({'Results': body})

def tail(file, n=1, bs=1024):
    f = open(file)
    f.seek(0, 2)
    l = 1 - f.read(1).count('\n')
    B = f.tell()
    while n >= l and B > 0:
      block = min(bs, B)
      B -= block
      f.seek(B, 0)
      l += f.read(block).count('\n')
    f.seek(B, 0)
    l = min(l, n)
    lines = f.readlines()[-l:]
    f.close()
    return lines

class LogHandler(tornado.web.RequestHandler):
  def get(self):
    # body = tornado.escape.json_decode(self.request.body)
    print('-----logger request-----')
    # print(body)
    # response = body
    # print(response)
    # logs_output = read_ta2_logs()
    lines = tail("ta2serverlog.log", 10000)

    logs_output = ('\n'.join(lines))
    self.write({'id':'200', 'logs_output':logs_output})
    # run_ta2_email(response)

  def post(self):
    pass

  def set_default_headers(self, *args, **kwargs):
    self.set_header("Access-Control-Allow-Origin", "*")
    self.set_header("Access-Control-Allow-Headers", "x-requested-with")
    self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/getlogger", LogHandler)
    ])

if __name__ == "__main__":
    # Logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='usrserverlog.log', filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        level=logging.INFO)

    # Evaluate
    app = make_app()
    app.listen(port=10235, address='ased-1.lti.cs.cmu.edu')
    tornado.ioloop.IOLoop.current().start()

