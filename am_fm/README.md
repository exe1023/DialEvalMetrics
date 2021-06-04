## The Deep AM-FM Framework

This framework intends to serve as a general evaluation framework for natural language generation tasks, such as machine translation, dialogue system and summarization. 

### Adequacy Metric

This component aims to assess the semantic aspect of system responses, more specifically, how much source information is preserved by the dialogue generation with reference to human-written responses. The continuous space model is adopted for evaluating adequacy where good word-level or sentence-level embedding techniques are studied to measure the semantic closessness of system responses and human references in the continous vector space.

### Fluency Metric

This component aims to assess the syntactic validity of system responses. It tries to compare the system hypotheses against human references in terms of their respective sentence-level normalized log probabilities based on the assumption that sentences that are of similar syntactic validity should share similar perplexity level given by a language model. Hence, in this component, various language model techniques are explored to accurately estimate the sentence-level probability distribution.

### Toolkit Requirements

1. python 3.x
2. emoji=0.5.4
3. jsonlines=1.2.0
4. tensorflow-gpu=1.14.0
5. tqdm=4.38.0

### Examples

Please refer to the example folder for detailed experimental setup and implementation steps of various evaluation tasks.

## References
<a id="1">[1]</a> 
Banchs, R. E., & Li, H. (2011, June). AM-FM: A semantic framework for translation quality assessment. In Proceedings of the 49th Annual Meeting of the ACL: Human Language Technologies: short papers-Volume 2 (pp. 153-158). ACL.
<br><br>
<a id="2">[2]</a>
 Banchs, R. E., D’Haro, L. F., & Li, H. (2015). Adequacy–fluency metrics: Evaluating MT in the continuous space model framework. IEEE/ACM TASLP, 23(3), 472-482.
<br><br>
<a id="3">[3]</a>
D'Haro, L. F., Banchs, R. E., Hori, C., & Li, H. (2019). Automatic evaluation of end-to-end dialog systems with adequacy-fluency metrics. Computer Speech & Language, 55, 200-215.

