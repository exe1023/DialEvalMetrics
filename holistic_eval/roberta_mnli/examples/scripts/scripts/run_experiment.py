import os
import smtplib
from email.mime.text import MIMEText
mail_host = 'smtp.163.com'
mail_user = 'aigu3525'
mail_pass = 'WOaibaobao'
sender = 'aigu3525@163.com'
receivers = ['liuyixian0317@gmail.com']


def run_training(ex_title, type, paras_dict, node, GPU, logger=None , print_=False):
    print('_'*100)
    if type == 'MNLI': train_file = 'run_glue.py'
    opt_dict = paras_dict
    try:
        os.mkdir('scripts/logs/' + type)
    except:
        x=1

    message = MIMEText('Start training experiment {}'.format(str(ex_title)), 'plain', 'utf-8')
    message['Subject'] = 'Experiment {}'.format(str(ex_title))
    message['From'] = sender
    message['To'] = receivers[0]
    try:
        smtpObj = smtplib.SMTP()
        smtpObj.connect(mail_host, 25)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(
            sender, receivers, message.as_string())
        smtpObj.quit()
        print('success')
    except:
        print('error')  # 打印错误

    if True:
        print_file_train = 'scripts/logs/'+ type + '/' + ex_title+ '.print'
        keys = list(opt_dict)
        values = [opt_dict[key] for key in keys]
        paras = ''
        for i in range(len(keys)):
            if values[i] == 'False':
                continue
            paras += ' --'
            paras += keys[i]
            if values[i] != 'True':
                paras += '='
                paras += str(values[i])
        if True:
            commend_list_train = []
            # print(paras)
            commend_list_train.append('ssh node'+node + ' \"')
            commend_list_train.append('cd /root/liuyx/transformers/examples;')
            commend_list_train.append('CUDA_VISIBLE_DEVICES=' + str(GPU) + ' /root/anaconda3/envs/transformer/bin/python ')
            commend_list_train.append(train_file + paras +' 2>&1 | tee '+print_file_train + '')
            commend_list_train.append('\"')
            print(commend_list_train)
            pred_return = os.system(''.join(commend_list_train))



        message = MIMEText('Experiment {}, training end'.format(str(ex_title)), 'plain', 'utf-8')
        message['Subject'] = 'Experiment {}'.format(str(ex_title))
        message['From'] = sender
        message['To'] = receivers[0]
        try:
            smtpObj = smtplib.SMTP()
            smtpObj.connect(mail_host, 25)
            smtpObj.login(mail_user, mail_pass)
            smtpObj.sendmail(
                sender, receivers, message.as_string())
            smtpObj.quit()
            print('success')
        except:
            print('error')  # 打印错误