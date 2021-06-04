wget http://yanran.li/files/ijcnlp_dailydialog.zip
unzip ijcnlp_dailydialog.zip
mv ijcnlp_dailydialog/ dailydialog/

cd ./dailydialog
unzip train.zip
unzip validation.zip
unzip test.zip
rm train.zip
rm validation.zip
rm test.zip