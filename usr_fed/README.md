# Dialogue Metrics Docker

The repository of the dockerized version of [Unsupervised Evaluation of Interactive Dialog with DialoGPT](https://github.com/Shikib/fed) and [USR: An Unsupervised and Reference Free Evaluation Metric for DialogGeneration](https://github.com/Shikib/usr).

Please refer to `fed` and `usr` two directories to see how to run the docker server on your machine.

Explanation of FED and USR: https://docs.google.com/document/d/14TsCe6Ih--Sai65RrUR05h054lod-PK4aw7gPxZ4QEo/edit?usp=sharing

## Local Testing
```
sh test_server.sh TESTFILE PORT
```

The script will send the test file to `http://localhost:PORT`. Server will process it and return results in json format.

For example, you can run
```
sh test_server.sh test_data/sample.json 8888
```

## API Formats

Input: (Sample is also in test_data/sample.json)
```
[
    { 
        "dialogid" : "xxx", 
        "system_name" : "xxx", 
        "date": "optional",
        "truncate_type": "normal",  
        "dialogue_context":  
        [ 
            {"agent": "scmamer", "text": "Hi!"}, 
            {"agent": "victim", "text": "Hello, how is your day?"}, 
            {"agent": "scmamer", "text": "Its good. Its raining a bit, but I am enjoying a good book. How about you?"} 
        ], 
        "response_list":  ["Its good, I just got back from walking my dog What book did you read?", "test", "test2"], 
        "agent_name": "victim" 
    },
    {
        "dialogid": "..."
        ....
    }
]
```

(11/18 update) 
Add "truncate_type" for the tradeoff between inference speed and (theoretical) performance.
- If "normal", then use the original version (batch size=2, max sequence length=128).
- If "no_truncate", then we don't do additional truncation but inference with cpu.
- If "more", then truncate each sentence more but use larger batch size (batch size=4, max sequence size=64)

Server Output: (take usr as an example)
```
{"Results":
    [
        { 
            "dialogid" : "xxx", 
            "system_name" : "xxx", 
            "date": "optional", 
            "dialogue_context":  
            [ 
                {"agent": "scmamer", "text": "Hi!"}, 
                {"agent": "victim", "text": "Hello, how is your day?"}, 
                {"agent": "scmamer", "text": "Its good. Its raining a bit, but I am enjoying a good book. How about you?"} 
            ], 
            "response_list":  ["Its good, I just got back from walking my dog What book did you read?", "test", "test2"], 
            "agent_name": "victim" ,
            "response_scores": [{"USR-DRc": 0.9966729879379272, "USR-DRf": 0.986003577709198, "USR-MLM": -2.6705663204193115, "USR": -0.07881129053731728}, {"USR-DRc": 0.007081418763846159, "USR-DRf": 0.14744246006011963, "USR-MLM": -4.4570112228393555, "USR": -10.832083514830789}, {"USR-DRc": 0.7456651926040649, "USR-DRf": 0.09365221858024597, "USR-MLM": -4.453603267669678, "USR": -7.29395564151886}]}]} # scores corresponding to each response
        }
        {
            "dialogid" : "...",
            ...
            "response_scores": [...]
        }
    ]
}
```


## Query Evaluationb Server

FED metric server is listening to `ased-1.lti.cs.cmu.edu:10234`

USR metric server is listening to `ased-1.lti.cs.cmu.edu:10235`

Example script for querying FED server:

```
curl --header "Content-Type: application/json" \
  --request POST \
  -d @test_data/sample.json \
  http://ased-1.lti.cs.cmu.edu:10234/
```

## Code Structure

```
.
├── fed (code for dockerized Unsupervised Evaluation of Interactive Dialog with DialoGPT)
│
├── usr (code for dockerized USR: An Unsupervised and Reference Free Evaluation Metric for DialogGeneration)   
│   
├── test_data (test data goes here)
│   ├── all_dialog.json (simply make AllDialogs.json downloed from lighthouse.csl.sri.com/~porras/EDM to our input format)
│   ├── all_dialog_clean.json (use dialog_parser.py to do some basic data cleaning on AllDialogs.json)
│   ├── sample.json (a sample input file, good for unit testing)
│   ├── dialog_parser.py (a dirty script to transform AllDialogs.json to our input format)
│   
└── results (test results goes here)
```

## Known Problems (9/27 Updated):

1. ~~Slow running speed of fed: In the original implementation, authors only implement the inference for single instance and there is no batch inference for the fed.~~ -> I have implemented batch inference for FED. Now the performance bottleneck lie on gpu memory size.
2. Weird score of USR-DRc and USR-DRf two metrics in usr: Currently I have not identiefied the cause of this problem, but have some guesses. In original paper, authors use `fact` with dialogue context to train RoBERTa on retrieval tasks to score responses. Here since we have no `fact` in our application, I simply copy the dialogue context as `fact` and use it as input. Another guess is it is due to the very long input / output length. Since it is the retrieval-based model, it is possible that the model gives very high score to long response as it may include more related information. -> I have discussed with authors of USR metrics. They told me USR metrics is desgined as self-supervised metrics but not for zero-shot evaluation. Thus it is reasonable that sometimes it gives weird results on our dataset. If we want to use it, we need to finetune it using our dialogues. However, currently we have few dialogues for training / development. We are still figuring our how to overcome this problem.
