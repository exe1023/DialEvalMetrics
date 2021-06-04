## PONE: A Feasible and Credible Automatic Evaluation of Open-domain Dialogue Systems with Enhanced Positive and Negative samples.

### 1. Framework

* Positive augmentor
* Label Filter
* Negative sampler
    * Fluency
    * Coherence
    * Safety
    
The motivation of our framework is to alleviate the extremely unbalanced data distribution of the negative sampling algorithm during training the score model of the learning-based metric.
In order to address this issue, from the aspect of providing enhanced positive samples and valuable negative samples, we propose a general and powerful evaluation framework. 

### 2. Experiment

#### 2.1 Xiaohuangji corpus
<table>
  <tr>
    <th colspan="2" rowspan="2">Metric</th>
    <th colspan="2">Fluency</th>
    <th colspan="2">Coherence</th>
    <th colspan="2">Engagement</th>
    <th colspan="2">Overall</th>
  </tr>
  <tr>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
  </tr>
  <tr>
    <td rowspan="2">Human annotator</td>
    <td>Human(avg)</td>
    <td>0.51019 (0.0)</td>
    <td>0.48081 (0.0)</td>
    <td>0.72133 (0.0)</td>
    <td>0.72947 (0.0)</td>
    <td>0.41929(0.0)</td>
    <td>0.42844(0.0)</td>
    <td>0.75171 (0.0)</td>
    <td>0.74741 (0.0)</td>
  </tr>
  <tr>
    <td>Human(max)</td>
    <td>0.61716 (0.0)</td>
    <td>0.61999 (0.0)</td>
    <td>0.77200 (0.0)</td>
    <td>0.77763 (0.0)</td>
    <td>0.57192 (0.0)</td>
    <td>0.56499 (0.0)</td>
    <td>0.85324 (0.0)</td>
    <td>0.84510 (0.0)</td>
  </tr>
  <tr>
    <td rowspan="7">Word-overlap-based</td>
    <td>BLEU-1</td>
    <td>0.08946 (0.12207)</td>
    <td>0.12362 (0.03232)</td>
    <td>0.07303 (0.20720)</td>
    <td>0.07642 (0.18684)</td>
    <td>0.06942 (0.23059)</td>
    <td>0.11981 (0.03808)</td>
    <td>0.06055 (0.29585)</td>
    <td>0.11981 (0.03808)</td>
  </tr>
  <tr>
    <td>BLEU-2</td>
    <td>0.13659 (0.01793)</td>
    <td>0.13004 (0.02429)</td>
    <td>0.08671 (0.13401)</td>
    <td>0.08036 (0.16502)</td>
    <td>0.13754 (0.01714)</td>
    <td>0.12141 (0.03556)</td>
    <td>0.09561 (0.09836)</td>
    <td>0.07666 (0.18545)</td>
  </tr>
  <tr>
    <td>BLEU-3</td>
    <td>0.14949 (0.00952)</td>
    <td>0.14317 (0.01306)</td>
    <td>0.08335 (0.14983)</td>
    <td>0.08087 (0.16235)</td>
    <td>0.15841 (0.00597)</td>
    <td>0.16307 (0.00463)</td>
    <td>0.09959 (0.08506)</td>
    <td>0.08820 (0.12743)</td>
  </tr>
  <tr>
    <td>BLEU-4</td>
    <td>0.15417 (0.00747)</td>
    <td>0.16685 (0.00375)</td>
    <td>0.08034 (0.16514)</td>
    <td>0.08256 (0.15374)</td>
    <td>0.1701 (0.00312)</td>
    <td>0.17861 (0.0019)</td>
    <td>0.09974 (0.08458)</td>
    <td>0.09149 (0.11380)</td>
  </tr>
  <tr>
    <td>ROUGE</td>
    <td>0.04684 (0.41885)</td>
    <td>0.06981 (0.22801)</td>
    <td>0.04336 (0.45436)</td>
    <td>0.06207 (0.28392)</td>
    <td>0.01929 (0.73945)</td>
    <td>0.03598 (0.53479)</td>
    <td>0.03787 (0.51346)</td>
    <td>0.05455 (0.34639)</td>
  </tr>
  <tr>
    <td>METEOR</td>
    <td>0.04125 (0.55134)</td>
    <td>0.31367 (0.0001)</td>
    <td>0.07639 (045003)</td>
    <td>0.38017 (0.0001)</td>
    <td>0.06886 (0.49601)</td>
    <td>0.38156 (9e-5)</td>
    <td>0.04199 (0.67826)</td>
    <td>0.4138 (2e-5)</td>
  </tr>
  <tr>
    <td>CIDEr</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="4">Embedding-based</td>
    <td>VE cosine</td>
    <td>0.19611 (0.00064)</td>
    <td>0.21638 (0.00016)</td>
    <td>0.11709 (0.04270)</td>
    <td>0.13326 (0.02096)</td>
    <td>0.26735 (0.0)</td>
    <td>0.29423 (0.0)</td>
    <td>0.16547 (0.00406)</td>
    <td>0.14981 (0.00936)</td>
  </tr>
  <tr>
    <td>EA cosine</td>
    <td>0.23591 (4e-5)</td>
    <td>0.25566 (1e-5)</td>
    <td>0.12451 (0.03108)</td>
    <td>0.15104 (0.00879)</td>
    <td>0.18031 (0.00171)</td>
    <td>0.17311 (0.00263)</td>
    <td>0.16708 (0.00370)</td>
    <td>0.16733 (0.00365)</td>
  </tr>
  <tr>
    <td>Gready Match</td>
    <td>0.20675 (0.00031)</td>
    <td>0.24259 (2e-5)</td>
    <td>0.09986 (0.08423)</td>
    <td>0.12652 (0.02845)</td>
    <td>0.16656 (0.00381)</td>
    <td>0.17898 (0.00186)</td>
    <td>0.12364 (0.03229)</td>
    <td>0.14015 (0.01513)</td>
  </tr>
  <tr>
    <td>BERTScore</td>
    <td>0.1972 (0.04924)</td>
    <td>0.35143 (0.00034)</td>
    <td>0.29753 (0.00264)</td>
    <td>0.39722 (4e-5)</td>
    <td>0.26698 (0.00725)</td>
    <td>0.37922 (4e-5)</td>
    <td>0.24294 (0.00725)</td>
    <td>0.37955 (0.0001)</td>
  </tr>
  <tr>
    <td rowspan="2">RUBER</td>
    <td>RUBER</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>BERT-RUBER</td>
    <td>0.15176 (0.13239)</td>
    <td>0.16293 (0.07129)</td>
    <td>0.42038 (0.0)</td>
    <td>0.43459 (0.0)</td>
    <td>0.287 (0.0)</td>
    <td>0.32058 (0.0)</td>
    <td>0.40056 (0.0)</td>
    <td>0.41203 (0.0)</td>
  </tr>
  <tr>
    <td rowspan="4">PoNe</td>
    <td>Negative</td>
    <td>0.15043 (0.06934)</td>
    <td>0.16143 (0.02147)</td>
    <td>0.45945 (0.0)</td>
    <td>0.46427 (0.0)</td>
    <td>0.2812 (0.0)</td>
    <td>0.30481 (0.0)</td>
    <td>0.42273 (0.0)</td>
    <td>0.44186 (0.0)</td>
  </tr>
  <tr>
    <td>Positive</td>
    <td>0.26901 (0.00961)</td>
    <td>0.3312 (0.00113)</td>
    <td>0.4619 (1e-5)</td>
    <td>0.52507 (0.0)</td>
    <td>0.36097 (0.00037)</td>
    <td>0.39595 (8e-5)</td>
    <td>0.43256 (2e-5)</td>
    <td>0.48417 (0.0)</td>
  </tr>
  <tr>
    <td>Positve+LF</td>
    <td>0.27248 (0.00731)</td>
    <td>0.31965 (0.00169)</td>
    <td>0.45781 (1e-5)</td>
    <td>0.52493 (0.0)</td>
    <td>0.37272 (0.00022)</td>
    <td>0.40252 (6e-5)</td>
    <td>0.43571 (2e-5)</td>
    <td>0.49291 (0.0)</td>
  </tr>
  <tr>
    <td>PoNe</td>
    <td>0.26585 (0.00975)</td>
    <td>0.32549 (0.00134)</td>
    <td>0.4620 (1e-5)</td>
    <td>0.52176 (0.0)</td>
    <td>0.36121 (0.00035)</td>
    <td>0.39333 (9e-5)</td>
    <td>0.43379 (2e-5)</td>
    <td>0.48248 (6e-5)</td>
  </tr>
</table>

#### 2.2 Tencent corpus
<table>
  <tr>
    <th colspan="2" rowspan="2">Metric</th>
    <th colspan="2">Fluency</th>
    <th colspan="2">Coherence</th>
    <th colspan="2">Engagement</th>
    <th colspan="2">Overall</th>
  </tr>
  <tr>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
  </tr>
  <tr>
    <td rowspan="2">Human annotator</td>
    <td>Human(avg)</td>
    <td>0.59136 (0.0)</td>
    <td>0.55898 (0.0)</td>
    <td>0.69134 (0.0)</td>
    <td>0.70729 (0.0)</td>
    <td>0.35185 (0.00111)</td>
    <td>0.35459 (0.0008)</td>
    <td>0.68023 (0.0)</td>
    <td>0.69423 (0.0)</td>
  </tr>
  <tr>
    <td>Human(max)</td>
    <td>0.61002 (0.0)</td>
    <td>0.57820 (0.0)</td>
    <td>0.74267 (0.0)</td>
    <td>0.75769 (0.0)</td>
    <td>0.65011 (0.0)</td>
    <td>0.65396 (0.0)</td>
    <td>0.74559 (0.0)</td>
    <td>0.76104 (0.0)</td>
  </tr>
  <tr>
    <td rowspan="7">Word-overlap-based</td>
    <td>BLEU-1</td>
    <td>0.01844 (0.75040)</td>
    <td>0.04806 (0.40685)</td>
    <td>-0.09169 (0.11300)</td>
    <td>-0.05471 (0.34498)</td>
    <td>0.01992 (0.01015)</td>
    <td>0.10524 (0.34498)</td>
    <td>-0.04804 (0.40702)</td>
    <td>-0.00437 (0.93998)</td>
  </tr>
  <tr>
    <td>BLEU-2</td>
    <td>0.04319 (0.45610)</td>
    <td>0.04609 (0.42635)</td>
    <td>-0.04600 (0.42728)</td>
    <td>-0.04348 (0.45308)</td>
    <td>0.14929 (0.00961)</td>
    <td>0.13134 (0.02289)</td>
    <td>-0.00994 (0.86382)</td>
    <td>0.00676 (0.90724)</td>
  </tr>
  <tr>
    <td>BLEU-3</td>
    <td>0.05464 (0.34560)</td>
    <td>0.04818 (0.40566)</td>
    <td>-0.02395 (0.67946)</td>
    <td>-0.03122 (0.59013)</td>
    <td>0.18843 (0.00133)</td>
    <td>0.18068 (0.00168)</td>
    <td>0.00954 (0.86936)</td>
    <td>0.01216 (0.83385)</td>
  </tr>
  <tr>
    <td>BLEU-4</td>
    <td>0.06013 (0.29924)</td>
    <td>0.05483 (0.34392)</td>
    <td>-0.01327 (0.81898)</td>
    <td>-0.02102 (0.71696)</td>
    <td>0.19803 (0.00056)</td>
    <td>0.20739 (0.0003)</td>
    <td>0.01899 (0.74321)</td>
    <td>0.02199 (0.00030)</td>
  </tr>
  <tr>
    <td>ROUGE</td>
    <td>0.15856 (0.00592)</td>
    <td>0.18726 (0.00112)</td>
    <td>0.11680 (0.04323)</td>
    <td>0.14720 (0.01068)</td>
    <td>0.13522 (0.01913)</td>
    <td>0.14914 (0.00968)</td>
    <td>0.13954 (0.01557)</td>
    <td>0.16717 (0.00369)</td>
  </tr>
  <tr>
    <td>METEOR</td>
    <td>0.30679 (0.00191)</td>
    <td>0.3157 (0.00138)</td>
    <td>0.27597 (0.00545)</td>
    <td>0.29124(0.00328)</td>
    <td>0.31405 (0.00146)</td>
    <td>0.31319 (0.00151)</td>
    <td>0.30146 (0.0023)</td>
    <td>0.2792 (0.00491)</td>
  </tr>
  <tr>
    <td>CIDEr</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="5">Embedding-based</td>
  <tr>
    <td>VE cosine</td>
    <td>0.01801 (0.75602)</td>
    <td>0.03729 (0.51996)</td>
    <td>0.00015 (0.99797)</td>
    <td>0.04825 (0.40499)</td>
    <td>0.28265 (0.0)</td>
    <td>0.36351 (0.0)</td>
    <td>0.03312 (0.56771)</td>
    <td>0.08603 (0.13711)</td>
  </tr>
  <tr>
    <td>EA cosine</td>
    <td>0.04536 (0.43373)</td>
    <td>0.01488 (0.79740)</td>
    <td>0.12448 (0.03113)</td>
    <td>0.08710 (0.13227)</td>
    <td>0.20538 (0.00033)</td>
    <td>0.17487 (0.00237)</td>
    <td>0.09581 (0.09766)</td>
    <td>0.07077 (0.22166)</td>
  </tr>
  <tr>
    <td>Gready Match</td>
    <td>0.05850 (0.31252)</td>
    <td>0.05752 (0.32070)</td>
    <td>0.11611 (0.04448)</td>
    <td>0.11414 (0.04824)</td>
    <td>0.22014 (0.00012)</td>
    <td>0.19262 (0.0008)</td>
    <td>0.12362 (0.03233)</td>
    <td>0.12046 (0.03704)</td>
  </tr>
  <tr>
    <td>BERTScore</td>
    <td>0.5057 (0.0)</td>
    <td>0.30529 (0.00201)</td>
    <td>0.37673 (0.00011)</td>
    <td>0.32128 (0.00112)</td>
    <td>0.34353 (0.00047)</td>
    <td>0.31939 (0.0012)</td>
    <td>0.40112 (4e-5)</td>
    <td>0.32937 (0.00082)</td>
  </tr>
  <tr>
    <td rowspan="2">RUBER</td>
    <td>RUBER</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>BERT-RUBER</td>
    <td>0.18071 (0.00209)</td>
    <td>0.18288 (0.00220)</td>
    <td>0.46455 (0.0)</td>
    <td>0.47206 (0.0)</td>
    <td>0.47028 (0.0)</td>
    <td>0.46026 (0.0)</td>
    <td>0.44231 (0.0)</td>
    <td>0.45003 (0.0)</td>
  </tr>
  <tr>
    <td rowspan="4">PoNe</td>
    <td>Negative</td>
    <td>0.29671 (0.00371)</td>
    <td>0.2952 (0.00378)</td>
    <td>0.51136 (0.0)</td>
    <td>0.49458 (0.0)</td>
    <td>0.48483 (1e-5)</td>
    <td>0.46738 (1e-5)</td>
    <td>0.4686 (2e-5)</td>
    <td>0.45029 (2e-5)</td>
  </tr>
  <tr>
    <td>Positive</td>
    <td>0.3344 (0.00108)</td>
    <td>0.32375 (0.00124)</td>
    <td>0.47329 (4e-5)</td>
    <td>0.47926 (0.0)</td>
    <td>0.35083 (0.00055)</td>
    <td>0.45327 (2e-5)</td>
    <td>0.44201 (1e-4)</td>
    <td>0.44292 (2e-5)</td>
  </tr>
  <tr>
    <td>Positive+LF</td>
    <td>0.30167 (0.0052)</td>
    <td>0.28417 (0.00533)</td>
    <td>0.47968 (3e-5)</td>
    <td>0.48227 (0.0)</td>
    <td>0.44216 (7e-5)</td>
    <td>0.44054 (2e-5)</td>
    <td>0.43785 (8e-5)</td>
    <td>0.43468 (2e-5)</td>
  </tr>
  <tr>
    <td>PoNe</td>
    <td>0.30542 (0.004)</td>
    <td>0.29794 (0.00314)</td>
    <td>0.47922 (3e-5)</td>
    <td>0.47539 (0.0)</td>
    <td>0.44572 (6e-5)</td>
    <td>0.43969 (3e-5)</td>
    <td>0.43896 (7e-5)</td>
    <td>0.43193 (3e-5)</td>
  </tr>
</table>


#### 2.3 Dailydialog corpus
<table>
  <tr>
    <th colspan="2" rowspan="2">Metric</th>
    <th colspan="2">Fluency</th>
    <th colspan="2">Coherence</th>
    <th colspan="2">Engaging</th>
    <th colspan="2">Overall</th>
  </tr>
  <tr>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
  </tr>
  <tr>
    <td rowspan="2">Human annotator</td>
    <td>Human(avg)</td>
    <td>0.25671 (0.04433)</td>
    <td>0.23077 (0.04018)</td>
    <td>0.58819 (0.0)</td>
    <td>0.56235 (0.0)</td>
    <td>0.17629 (0.47127)</td>
    <td>0.17013 (0.48079)</td>
    <td>0.30882 (0.08065)</td>
    <td>0.30775 (0.0817)</td>
  </tr>
  <tr>
    <td>Human(max)</td>
    <td>0.35980 (0.12222)</td>
    <td>0.28559 (0.10125)</td>
    <td>0.71379 (0.0)</td>
    <td>0.68923 (0.0)</td>
    <td>0.44772 (0.96780)</td>
    <td>0.43289 (0.99546)</td>
    <td>0.59791 (0.19271)</td>
    <td>0.59092 (0.20339)</td>
  </tr>
  <tr>
    <td rowspan="7">Word-overlap-based</td>
    <td>BLEU-1</td>
    <td>0.16335 (0.10397)</td>
    <td>0.10365 (0.30478)</td>
    <td>0.00471 (0.51825)</td>
    <td>0.02984 (0.65568)</td>
    <td>-0.0345 (0.07609)</td>
    <td>-0.02819 (0.0582)</td>
    <td>0.05075 (0.2360)</td>
    <td>0.04072 (0.32693)</td>
  </tr>
  <tr>
    <td>BLEU-2</td>
    <td>0.14949 (0.15892)</td>
    <td>0.04786 (0.6363)</td>
    <td>-0.03278 (0.16866)</td>
    <td>0.03422 (0.63277)</td>
    <td>0.09611 (0.34147)</td>
    <td>0.02878 (0.16029)</td>
    <td>-0.05363 (0.1829)</td>
    <td>0.0259 (0.16739)</td>
  </tr>
  <tr>
    <td>BLEU-3</td>
    <td>0.12854 (0.20249)</td>
    <td>0.00941 (0.8566)</td>
    <td>-0.0502 (0.9956)</td>
    <td>0.01907 (0.52688)</td>
    <td>0.13893 (0.16805)</td>
    <td>0.07762 (0.31379)</td>
    <td>-0.09575 (0.12640)</td>
    <td>-0.00196 (0.14129)</td>
  </tr>
  <tr>
    <td>BLEU-4</td>
    <td>0.12195 (0.22677)</td>
    <td>-0.0206 (0.6924)</td>
    <td>-0.05859 (0.07712)</td>
    <td>0.01682 (0.45016)</td>
    <td>0.1567 (0.11948)</td>
    <td>0.13136 (0.19268)</td>
    <td>-0.09711 (0.10709)</td>
    <td>-0.02268 (0.18998)</td>
  </tr>
  <tr>
    <td>ROUGE</td>
    <td>0.22758 (0.02277)</td>
    <td>0.16664 (0.09751)</td>
    <td>0.29557 (0.00283)</td>
    <td>0.20947 (0.03647)</td>
    <td>0.22673 (0.0233)</td>
    <td>0.11552 (0.25242)</td>
    <td>0.25111 (0.01173)</td>
    <td>0.19014 (0.05811)</td>
  </tr>
  <tr>
    <td>METEOR</td>
    <td>0.21733 (0.02986)</td>
    <td>0.15244 (0.13001)</td>
    <td>0.29886 (0.00252)</td>
    <td>0.18758 (0.06164)</td>
    <td>0.26144 (0.0086)</td>
    <td>0.17211 (0.08684)</td>
    <td>0.23501 (0.01859)</td>
    <td>0.17322 (0.08481)</td>
  </tr>
  <tr>
    <td>CIDEr</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="4">Embedding-based</td>
    <td>EA cosine</td>
    <td>0.27328 (0.00594)</td>
    <td>0.25212 (0.01139)</td>
    <td>0.37456 (0.00012)</td>
    <td>0.34373 (0.00046)</td>
    <td>0.28161 (0.00453)</td>
    <td>0.25026 (0.01203)</td>
    <td>0.34867 (0.00038)</td>
    <td>0.31868 (0.00123)</td>
  </tr>
  <tr>
    <td>VE cosine</td>
    <td>0.16257 (0.01954)</td>
    <td>0.17511 (0.0814)</td>
    <td>0.41057 (2e-05)</td>
    <td>0.39348 (5e-5)</td>
    <td>0.20061 (0.04536)</td>
    <td>0.19938 (0.04673)</td>
    <td>0.34634 (0.00042)</td>
    <td>0.37443 (0.00012)</td>
  </tr>
  <tr>
    <td>Gready Match</td>
    <td>0.11472 (0.20558)</td>
    <td>0.06597 (0.27803)</td>
    <td>0.18876 (0.06)</td>
    <td>0.12979 (0.19808)</td>
    <td>0.19689 (0.04959)</td>
    <td>0.14733 (0.14354)</td>
    <td>0.25477 (0.01053)</td>
    <td>0.23818 (0.01702)</td>
  </tr>
  <tr>
    <td>BERTScore</td>
    <td>0.49783 (0.0)</td>
    <td>0.19963 (0.04645)</td>
    <td>0.23521 (0.01849)</td>
    <td>0.33128 (0.00076)</td>
    <td>0.28702 (0.00379)</td>
    <td>0.16085 (0.10988)</td>
    <td>0.20493 (0.04083)</td>
    <td>0.22119 (0.027)</td>
  </tr>
  <tr>
    <td rowspan="2">RUBER</td>
    <td>RUBER</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>BERT-RUBER</td>
    <td>0.35603 (0.00075)</td>
    <td>0.38392 (0.00019)</td>
    <td>0.37732 (0.00086)</td>
    <td>0.41318 (0.00011)</td>
    <td>0.35303 (0.00133)</td>
    <td>0.38185 (0.00029)</td>
    <td>0.40004 (0.00021)</td>
    <td>0.42467 (6e-05)</td>
  </tr>
  <tr>
    <td rowspan="4">PoNe</td>
    <td>Negative</td>
    <td>0.34965 (0.00074) </td>
    <td>0.37084 (0.00034)</td>
    <td>0.39973 (0.00016)</td>
    <td>0.42562 (7e-5)</td>
    <td>0.37437 (0.00086)</td>
    <td>0.3891 (0.0008)</td>
    <td>0.41684 (4e-5)</td>
    <td>0.42858 (3e-5)</td>
  </tr>
  <tr>
    <td>Positive</td>
    <td>0.38986 (0.0003)</td>
    <td>0.43225 (2e-5)</td>
    <td>0.40301 (8e-5)</td>
    <td>0.45635 (1e-5)</td>
    <td>0.38052 (0.00025)</td>
    <td>0.42091 (3e-5)</td>
    <td>0.46771 (3e-5)</td>
    <td>0.47485 (1e-5)</td>
  </tr>
  <tr>
    <td>Positive+LF</td>
    <td>0.45793 (1e-5)</td>
    <td>0.47173 (1e-5)</td>
    <td>0.47338 (0.0)</td>
    <td>0.50341 (0.0)</td>
    <td>0.38936 (0.00027)</td>
    <td>0.36527 (0.00093)</td>
    <td>0.49814 (0.0)</td>
    <td>0.50892 (0.0)</td>
  </tr>
  <tr>
    <td>PoNe</td>
    <td>0.43744 (1e-5)</td>
    <td>0.45176 (0.0)</td>
    <td>0.4517 (1e-5)</td>
    <td>0.48347 (0.0)</td>
    <td>0.42464 (3e-5)</td>
    <td>0.44577 (1e-5)</td>
    <td>0.49295 (0.0)</td>
    <td>0.49943 (0.0)</td>
  </tr>
</table>


#### 2.5 Cornell corpus

<table>
  <tr>
    <th colspan="2" rowspan="2">Metric</th>
    <th colspan="2">Fluency</th>
    <th colspan="2">Coherence</th>
    <th colspan="2">Engaging</th>
    <th colspan="2">Overall</th>
  </tr>
  <tr>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
    <td>Pearson (p-value)</td>
    <td>Spearman (p-value)</td>
  </tr>
  <tr>
    <td rowspan="2">Human annotator</td>
    <td>Human(avg)</td>
    <td>0.26473 (0.00817)</td>
    <td>0.28065 (0.00735)</td>
    <td>0.41958 (0.00012)</td>
    <td>0.42370 (0.00016)</td>
    <td>0.30206 (0.02393)</td>
    <td>0.30696 (0.0279)</td>
    <td>0.40861 (3e-5)</td>
    <td>0.39030 (9e-5)</td>
  </tr>
  <tr>
    <td>Human(max)</td>
    <td>0.27352 (0.01220)</td>
    <td>0.32928 (0.01426)</td>
    <td>0.52376 (0.00027)</td>
    <td>0.52120 (0.00044)</td>
    <td>0.47661 (0.05375)</td>
    <td>0.49765 (0.06732)</td>
    <td>0.42845 (6e-5)</td>
    <td>0.40727 (0.00019)</td>
  </tr>
  <tr>
    <td rowspan="7">Word-overlap-based</td>
    <td>BLEU-1</td>
    <td>0.18564 (0.06444)</td>
    <td>0.12662 (0.20935)</td>
    <td>0.29230 (0.00317)</td>
    <td>0.23243 (0.01996)</td>
    <td>0.13845 (0.16954)</td>
    <td>0.08662 (0.39148)</td>
    <td>0.23482 (0.01869)</td>
    <td>0.18540 (0.06478)</td>
  </tr>
  <tr>
    <td>BLEU-2</td>
    <td>0.14540 (0.14890)</td>
    <td>0.12720 (0.20726)</td>
    <td>0.25807 (0.00953)</td>
    <td>0.23426 (0.01898)</td>
    <td>0.10003 (0.32206)</td>
    <td>0.08776 (0.38526)</td>
    <td>0.20988 (0.03610)</td>
    <td>0.18732 (0.06202)</td>
  </tr>
  <tr>
    <td>BLEU-3</td>
    <td>0.12858 (0.20233)</td>
    <td>0.11713 (0.24581)</td>
    <td>0.24158 (0.01546)</td>
    <td>0.22289 (0.02581)</td>
    <td>0.08322 (0.41040)</td>
    <td>0.08048 (0.42604)</td>
    <td>0.19429 (0.05275)</td>
    <td>0.18069 (0.07201)</td>
  </tr>
  <tr>
    <td>BLEU-4</td>
    <td>0.23403 (0.01910)</td>
    <td>0.21574 (0.03111)</td>
    <td>0.07560 (0.45472)</td>
    <td>0.06413 (0.52618)</td>
    <td>0.07560 (0.45472)</td>
    <td>0.06413 (0.52618)</td>
    <td>0.18684 (0.06269)</td>
    <td>0.17036 (0.09015)</td>
  </tr>
  <tr>
    <td>ROUGE</td>
    <td>0.06574 (0.51580)</td>
    <td>0.08395 (0.40633)</td>
    <td>0.10050 (0.31980)</td>
    <td>0.07348 (0.46750)</td>
    <td>0.08570 (0.39654)</td>
    <td>0.08023 (0.42747)</td>
    <td>0.06357 (0.52977)</td>
    <td>0.04233 (0.67585)</td>
  </tr>
  <tr>
    <td>METEOR</td>
    <td>0.1718 (0.08742)</td>
    <td>0.21569 (0.03114)</td>
    <td>0.19591 (0.05077)</td>
    <td>0.29001 (0.00342)</td>
    <td>0.13889 (0.16815)</td>
    <td>0.33335 (0.0007)</td>
    <td>0.15578 (0.12171)</td>
    <td>0.226 (0.02377)</td>
  </tr>
  <tr>
    <td>CIDEr</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="4">Embedding-based</td>
    <td>EA cosine</td>
    <td>0.13707 (0.17387)</td>
    <td>0.17891 (0.07491)</td>
    <td>0.09043 (0.37089)</td>
    <td>0.10305 (0.30761)</td>
    <td>0.05457 (0.58972)</td>
    <td>0.05675 (0.57490)</td>
    <td>0.12218 (0.22592)</td>
    <td>0.12130 (0.22930)</td>
  </tr>
  <tr>
    <td>VE cosine</td>
    <td>0.12286 (0.22329)</td>
    <td>0.15819 (0.11598)</td>
    <td>0.08025 (0.42736)</td>
    <td>0.08180 (0.41846)</td>
    <td>-8e-5 (0.99940)</td>
    <td>0.00487 (0.96164)</td>
    <td>0.03397 (0.73722)</td>
    <td>0.03101 (0.75939)</td>
  </tr>
  <tr>
    <td>Gready Match</td>
    <td>0.16361 (0.10385)</td>
    <td>0.11768 (0.24359)</td>
    <td>0.08837 (0.38194)</td>
    <td>0.03244 (0.74866)</td>
    <td>-0.00064 (0.99496)</td>
    <td>-0.02827 (0.78009)</td>
    <td>0.05066 (0.61669)</td>
    <td>-0.01348 (0.89411)</td>
  </tr>
  <tr>
    <td>BERTScore</td>
    <td>0.30342 (0.00215)</td>
    <td>0.38671 (7e-5)</td>
    <td>0.29458 (0.00293)</td>
    <td>0.4154 (2e-5)</td>
    <td>0.2191 (0.02851)</td>
    <td>0.25292 (0.01112)</td>
    <td>0.26481 (0.00776)</td>
    <td>0.31448 (0.00114)</td>
  </tr>
  <tr>
    <td rowspan="2">RUBER</td>
    <td>RUBER</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>BERT-RUBER</td>
    <td>0.26292 (0.01603)</td>
    <td>0.28147 (0.00538)</td>
    <td>0.35468 (0.00271)</td>
    <td>0.33564 (0.00102)</td>
    <td>0.20349 (0.06535)</td>
    <td>0.24332 (0.02090)</td>
    <td>0.31929 (0.00335)</td>
    <td>0.30214 (0.00601)</td>
  <tr>
    <td rowspan="4">PoNe</td>
    <td>Negative</td>
    <td>0.24918 (0.0177)</td>
    <td>0.28347 (0.00595)</td>
    <td>0.3228 (0.0034)</td>
    <td>0.31731 (0.00206)</td>
    <td>0.19255 (0.07044)</td>
    <td>0.22445 (0.04023)</td>
    <td>0.28091 (0.01071)</td>
    <td>0.29461 (0.00485)</td>
  </tr>
  <tr>
    <td>Positive</td>
    <td>0.39201 (8e-5)</td>
    <td>0.43083 (1e-5)</td>
    <td>0.37165 (0.00023)</td>
    <td>0.41875 (4e-5)</td>
    <td>0.21547 (0.03355)</td>
    <td>0.24386 (0.01981)</td>
    <td>0.34594 (0.00066)</td>
    <td>0.34413 (0.00103)</td>
  </tr>
  <tr>
    <td>Positive+LF</td>
    <td>0.34835 (0.0007)</td>
    <td>0.36434 (0.00038)</td>
    <td>0.43794 (1e-5)</td>
    <td>0.44554 (1e-5)</td>
    <td>0.24109 (0.01953)</td>
    <td>0.26544 (0.00979)</td>
    <td>0.38447 (0.00017)</td>
    <td>0.36607 (0.00037)</td>
  </tr>
  <tr>
    <td>PoNe</td>
    <td>0.31293 (0.0026)</td>
    <td>0.31905 (0.00239)</td>
    <td>0.36452 (3e-4)</td>
    <td>0.37794 (2e-4)</td>
    <td>0.21996 (0.029)</td>
    <td>0.22896 (0.02278)</td>
    <td>0.34091 (8e-4)</td>
    <td>0.3316 (0.0014)</td>
  </tr>
</table>