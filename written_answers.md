# Key Files

- src/train.py
-- Main training script used for all four tasks. Argument flags in the CLI are used to set specific types of tests to be run.

- src/models/rnn_vanilla.py
-- Model for Vanilla RNN (used for both character and word-level)

- src/models/rnn_lstm.py
-- Model for LSTM RNN (used for both character and word-level).


# Task 1 - Vanilla RNN (Character-level)

## Initial Findings

For this model, I tried numerous different configurations and iterations of approaches. A complete list of the various tests run can be found in artifacts/results_summary.txt.

The model I built showed very promising results when working with large numbers of hidden networks (512) and large sequence lengths (200). Batch sizes had a relatively small impact.

I initially attempted to implement the vanilla RNN entirely in numpy but ran into performance issues and ultimately included PyTorch libraries to speed up the gradient calculations. To check gradients I included a
function which performs an autograd_sanity check to ensure that the autograd is producing realistic numbers for each test. These results are stored against each test in artifacts/logs/(name of test)/autograd_sanity.txt.

Specific settings from the most promising initial results are shown below (and are stored in artifacts/metadata). This specific test was cancelled after one epoch as the training time was unrealistically long.

{
  "exp_name": "final_h512_s150_b64",
  "mode": "char",
  "model_type": "rnn",
  "hidden": 512,
  "batch": 64,
  "seq": 150,
  "lr": 0.0005,
  "epochs": 12,
  "seed": 123,
  "created_at": "2026-04-19T02:58:00.776813+00:00"
}

**Sample Text:**

Two to stingur that they soon spirition to retaring while his waichners, particularly. It seized this in English,  Ned Land Island, and that we were, Rodamp Clam. This man bonher-ships and senses. These animals full, he was rapidly liuterysts. After we hadn t find it then culliss, in order with its culter and devoted for us to rearing, my friends, comparing the huge sleeping in a few was stretched

## Pivoting to Reduced Hyperparameters

At these sizes it was taking 2+ hours to run a single epoch so I determined to decrease the hyperparameters significantly in order to be able to compare the impact of hyperparameters on the results. The epoch lengths were still fairly long (at least 5-6 minutes with the simplest settings), so in the interest of time I ran these tests for 5 epochs and did breakpoints for each epoch. 

The tests I ran were combinations of:
- Hidden Layer: 32, 64
- Sequence Length: 25, 50

### Results from Hidden Layers: 32, Sequence Length: 25

![(plot)](artifacts/plots/base_h32_s25/loss_vs_epoch.png)

**Sample Text Epoch 1**
Tun clote whet the Capblarve, aplovers," Done its moond we critcipunite lungifulle of gougld was mides hunt it nof-ooved seak. The wheacty-nour and the doggssieving istery at this of but sappperserd fi

**Sample Text Epoch 2**
The sifing had rassour of thiscle to voulsa but, beriss foltionly oor out had be the slan werm the mary. On. We by to the anvasantly to mas of Arisod at moor work groundertatill, atress we scomplare, t

**Sample Text Epoch 3**
T-As. Thet brseacered some alcinqueggurned far a paings umbenders the I heving twoppent letired the Naved he wertrequenth of it, praptest, the way, frea; and ver. trike into thror was at stilaged, who

**Sample Text Epoch 4**
The houghtalize not of he repan becevelase, the bees tt wall piningrious with Mo a my thes sif, but in. I ans byse cammerfined recordes to my but outail suast ong lagent, stere lase to grous, but conge

**Sample Text Epoch 5**
Thay to caples a goalpanst, and mong. Waves nop becires. It formlice! Which. He shout a not is well outhy exquigisalt on hunclisute, Chackigite.  Crvein to exhrues tsung of them? We brund I fow pla

**Notes**
We saw loss start at 2.1453 and decrease over the five epochs to 1.9347. With these hyperparameters we're seeing text that is recognizable as the format of English words and sentences.
The first sample shows some English words (the, had, of, to, but, out, had, be, the, we, work). And as the epoches go on we see a greater number of recognizable English words as well
(it, which, he, shout, a, not, is, on, well, we, I), but all of the samples are largely incomprehensible.

### Results from Hidden Layers: 32, Sequence Length: 50

![alt text](artifacts/plots/base_h32_s50/loss_vs_epoch.png)

**Sample Text Epoch 1**
Trea were whwere righibly voays. .8! Chis Capt oce,  dandic itsioud thall bugut empachougliligu. We whuck it not-doved tear. The wheacty-you. It m know sainid Ne. hazater in the bunered tree digg

**Sample Text Epoch 2**
TOD NOAL THERE ARSSom. The liggll to conesan T I besils and in loge reowhing to have of the rile. The plang. Why to the ruvisher, himptroins, ais enate to demonygens." "Betall; atreped ons. Consula t

**Sample Text Epoch 3**
T-Proftent breat, beats walf, seated the cunde a passes ummepegred speed shipthings juscly freaid be sace he were equentiony. Cave a bust, the way scambery. Besioust. We into throalcas at stilagd itsth

**Sample Text Epoch 4**
T Nevoully oped not of shorpece beclowlase, and be extter onde." "Whate the trepeated thes Ther all that an this to deven ned recordss to aselul ourame suasm one vase under upon see town t some whill 

**Sample Text Epoch 5**
TEREN FASA SSRINDES ISKOUThe IR ApS.. Nyproing hazin; flow, formeptevions s dower. The saint." 3w, to unthy exquighesst on hance therg chick grent, brumin Neeest. It tsunly, quact? We loked I fule Fo

**Notes**
We saw loss start at 2.0168  and decrease over epochs to 1.90551. This is an improvement over our previous test. Similarly to the previous test, we see a format that is recognizable, and we do see
English words appearing throughout the samples. Specifically in the last sample we see: flow, saint, on, chick, it, we, I. But again the samples are still largely unrecognizable.

### Results from Hidden Layers: 64, Sequence Length: 25

![(plot)](artifacts/plots/base_h64_s25/loss_vs_epoch.png)

**Sample Text Epoch 1**
This cerience,  Candic its ruch an the 17730 passouglilian mides hund it now-doved teme. The whice pryse. I we m know ssining his stary-bridus, Capthere here-Raigilefuching the told his give a wors fea

**Sample Text Epoch 2**
There would lysis folting hoon mouth: kiff the our awery--I am appane. Well ontayor. Here curieft. They is enexthesed my you digented iffictrips were of threast furces End & Exiterven by Hemps his o

**Sample Text Epoch 3**
They had a passefick the stats? I hearprisw poors let a Six errays, he wertay that project praptes adgles!" "Lea hert. Besivert. We into throas with stilag there of arong chower. But run the lad; than

**Sample Text Epoch 4**
Tavery, ewbsee the wonded indrimute lo't said; and surflup our which theres to deven neil went was vaysed, to tame sung forgel genuous bugge see towar from the some thrugity! No woulded hact to firme

**Sample Text Epoch 5**
There stemperitered the welt eveors  he water out arifusisw, to unterrerm in salf--suspors therg concors so the peid regents us trangerating to did brood I went Faxt orcolamened firrs, evened to the wo

**Notes**
We saw loss begin at 2.0024 and decrease over epochs to 1.75035. This is a further improvement over the previous two tests. Again we see a recognizable format and and increasing number of actual English words, 
but the text is still not English and is incomprehensible.

### Results from Hidden Layers: 64, Sequence Length: 50

![(plot)](artifacts/plots/base_h64_s50/loss_vs_epoch.png)

**Sample Text Epoch 1**
This seas Fresther; Iceitain; Ats ling 17760 peerouglilial mides hund it not-loved temp. The whice you, that tomend. The Ciden is soow the rustly be back treemed then chith the toldint I inno end suffa

**Sample Text Epoch 2**
These burneners. Aftling, whom moung: kiff the old an told parce, and. We at at occumaraticurized bilones, of at the deminent stombermallf at mod when one treast fuoles End & Ebits, on back, Phoses o

**Sample Text Epoch 3**
The underwated from my dgresosposing into will it detiacineter ance he wert equention comple abads and bear gear here. Besiluptrizer Touthert--was attetilagination to recorch, morisestrun the ladeen, n

**Sample Text Epoch 4**
Tas and ears a valwarnd in her out ele's said; and 2  Para curriep ressible to delence in a vists to as but outaily underward, genuoustered Isk graghous, but chancus. THE CHAPTER 21 AT SES. METARDURTUR

**Sample Text Epoch 5**
TAN VOAY AIHASS FOGUG LESTAS ON RANCOREPTER THE CHAPTER 23 Leformuted exquights was flutters)? Concongry to betein of exhentives, like them? We burned the armed go to and vanicary, eved with the wo

**Notes**
We saw loss begin at 1.9319 and decrease over the epochs to 1.69868. Another improvement over the previous three tests. By the end of the last epoch the text is still incomprehensible but contains far
more recognizable words than in any previous test and is beginning to take shape.

### Overall Takeaways

As suggested by the initial tests, the training would benefit from a larger number of hidden layers and larger sequence lengths, which help to improve faster convergence over epochs. Larger numbers of epochs also clearly improve results. 

# Task 2 - LSTM RNN (Character-level)