curl --header "Content-Type: application/json" \
  --request POST \
  -d @$1 \
  http://ased-1.lti.cs.cmu.edu:$2/
