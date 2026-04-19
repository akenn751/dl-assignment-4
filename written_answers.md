# Key Files

- src/train.py
-- Main training script used for all four tasks. Argument flags in the CLI are used to set specific types of tests to be run.

- src/models/rnn_vanilla.py
-- Model for Vanilla RNN (used for both character and word-level)

- src/models/rnn_lstm.py
-- Model for LSTM RNN (used for both character and word-level).

# Example Command for Running train.py with arguments
python -m src.train --mode word --model rnn --epochs 5 --batch 32 --seq 25 --hidden 32 --lr 5e-4 --exp-name rnn_word_h32_s25 --save-dir artifacts --save-final-only --breakpoints 1,2,3,4,5 --sample-length 200 --
seed 123 --seed-text "The "


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

## Results from Hidden Layers: 32, Sequence Length: 25

![(plot)](artifacts/plots/base_h32_s25/loss_vs_epoch.png)

**Sample Text Epoch 1**\
Tun clote whet the Capblarve, aplovers," Done its moond we critcipunite lungifulle of gougld was mides hunt it nof-ooved seak. The wheacty-nour and the doggssieving istery at this of but sappperserd fi

**Sample Text Epoch 2**\
The sifing had rassour of thiscle to voulsa but, beriss foltionly oor out had be the slan werm the mary. On. We by to the anvasantly to mas of Arisod at moor work groundertatill, atress we scomplare, t

**Sample Text Epoch 3**\
T-As. Thet brseacered some alcinqueggurned far a paings umbenders the I heving twoppent letired the Naved he wertrequenth of it, praptest, the way, frea; and ver. trike into thror was at stilaged, who

**Sample Text Epoch 4**\
The houghtalize not of he repan becevelase, the bees tt wall piningrious with Mo a my thes sif, but in. I ans byse cammerfined recordes to my but outail suast ong lagent, stere lase to grous, but conge

**Sample Text Epoch 5**\
Thay to caples a goalpanst, and mong. Waves nop becires. It formlice! Which. He shout a not is well outhy exquigisalt on hunclisute, Chackigite.  Crvein to exhrues tsung of them? We brund I fow pla

**Notes**\
We saw loss start at 2.1453 and decrease over the five epochs to 1.9347. With these hyperparameters we're seeing text that is recognizable as the format of English words and sentences.
The first sample shows some English words (the, had, of, to, but, out, had, be, the, we, work). And as the epoches go on we see a greater number of recognizable English words as well
(it, which, he, shout, a, not, is, on, well, we, I), but all of the samples are largely incomprehensible.

## Results from Hidden Layers: 32, Sequence Length: 50

![alt text](artifacts/plots/base_h32_s50/loss_vs_epoch.png)

**Sample Text Epoch 1**\
Trea were whwere righibly voays. .8! Chis Capt oce,  dandic itsioud thall bugut empachougliligu. We whuck it not-doved tear. The wheacty-you. It m know sainid Ne. hazater in the bunered tree digg

**Sample Text Epoch 2**\
TOD NOAL THERE ARSSom. The liggll to conesan T I besils and in loge reowhing to have of the rile. The plang. Why to the ruvisher, himptroins, ais enate to demonygens." "Betall; atreped ons. Consula t

**Sample Text Epoch 3**\
T-Proftent breat, beats walf, seated the cunde a passes ummepegred speed shipthings juscly freaid be sace he were equentiony. Cave a bust, the way scambery. Besioust. We into throalcas at stilagd itsth

**Sample Text Epoch 4**\
T Nevoully oped not of shorpece beclowlase, and be extter onde." "Whate the trepeated thes Ther all that an this to deven ned recordss to aselul ourame suasm one vase under upon see town t some whill 

**Sample Text Epoch 5**\
TEREN FASA SSRINDES ISKOUThe IR ApS.. Nyproing hazin; flow, formeptevions s dower. The saint." 3w, to unthy exquighesst on hance therg chick grent, brumin Neeest. It tsunly, quact? We loked I fule Fo

**Notes**\
We saw loss start at 2.0168  and decrease over epochs to 1.90551. This is an improvement over our previous test. Similarly to the previous test, we see a format that is recognizable, and we do see
English words appearing throughout the samples. Specifically in the last sample we see: flow, saint, on, chick, it, we, I. But again the samples are still largely unrecognizable.

## Results from Hidden Layers: 64, Sequence Length: 25

![(plot)](artifacts/plots/base_h64_s25/loss_vs_epoch.png)

**Sample Text Epoch 1**\
This cerience,  Candic its ruch an the 17730 passouglilian mides hund it now-doved teme. The whice pryse. I we m know ssining his stary-bridus, Capthere here-Raigilefuching the told his give a wors fea

**Sample Text Epoch 2**\
There would lysis folting hoon mouth: kiff the our awery--I am appane. Well ontayor. Here curieft. They is enexthesed my you digented iffictrips were of threast furces End & Exiterven by Hemps his o

**Sample Text Epoch 3**\
They had a passefick the stats? I hearprisw poors let a Six errays, he wertay that project praptes adgles!" "Lea hert. Besivert. We into throas with stilag there of arong chower. But run the lad; than

**Sample Text Epoch 4**\
Tavery, ewbsee the wonded indrimute lo't said; and surflup our which theres to deven neil went was vaysed, to tame sung forgel genuous bugge see towar from the some thrugity! No woulded hact to firme

**Sample Text Epoch 5**\
There stemperitered the welt eveors  he water out arifusisw, to unterrerm in salf--suspors therg concors so the peid regents us trangerating to did brood I went Faxt orcolamened firrs, evened to the wo

**Notes**\
We saw loss begin at 2.0024 and decrease over epochs to 1.75035. This is a further improvement over the previous two tests. Again we see a recognizable format and and increasing number of actual English words, 
but the text is still not English and is incomprehensible.

## Results from Hidden Layers: 64, Sequence Length: 50

![(plot)](artifacts/plots/base_h64_s50/loss_vs_epoch.png)

**Sample Text Epoch 1**\
This seas Fresther; Iceitain; Ats ling 17760 peerouglilial mides hund it not-loved temp. The whice you, that tomend. The Ciden is soow the rustly be back treemed then chith the toldint I inno end suffa

**Sample Text Epoch 2**\
These burneners. Aftling, whom moung: kiff the old an told parce, and. We at at occumaraticurized bilones, of at the deminent stombermallf at mod when one treast fuoles End & Ebits, on back, Phoses o

**Sample Text Epoch 3**\
The underwated from my dgresosposing into will it detiacineter ance he wert equention comple abads and bear gear here. Besiluptrizer Touthert--was attetilagination to recorch, morisestrun the ladeen, n

**Sample Text Epoch 4**\
Tas and ears a valwarnd in her out ele's said; and 2  Para curriep ressible to delence in a vists to as but outaily underward, genuoustered Isk graghous, but chancus. THE CHAPTER 21 AT SES. METARDURTUR

**Sample Text Epoch 5**\
TAN VOAY AIHASS FOGUG LESTAS ON RANCOREPTER THE CHAPTER 23 Leformuted exquights was flutters)? Concongry to betein of exhentives, like them? We burned the armed go to and vanicary, eved with the wo

**Notes**\
We saw loss begin at 1.9319 and decrease over the epochs to 1.69868. Another improvement over the previous three tests. By the end of the last epoch the text is still incomprehensible but contains far
more recognizable words than in any previous test and is beginning to take shape.

## Overall Takeaways

As suggested by the initial tests, the training would benefit from a larger number of hidden layers and larger sequence lengths, which help to improve faster convergence over epochs. Larger numbers of epochs also clearly improve results. 

# Task 2 - LSTM RNN (Character-level)

## Experimental Setup

In order to more accurately compare across our tests, I used the same configurations as in Task 1, only adjusting the model used to LSTM, but preserving the same structure of hidden layers, sequence lengths, epochs, etc.

## Results from Hidden Layers: 32, Sequence Length: 25

![(plot)](artifacts/plots/lstm_char_h32_s25/loss_vs_epoch.png)

**Sample Text Epoch 1**\
The cried punstanly bigut empergouglilian mides hund it not-doved tempacate. That yey, that tome. Thens Cit Nevited water rustly bepers harse difallour in of you lin Bun in paendes feat or the bomterus

**Sample Text Epoch 2**\
The treply ook out haking the suddanes Ned THE Nag obly that capsool. He Abluct framing I is eneen to me.  Conseil at this said padient. Calame Fudy our prodentrests, on battemes his opperfom ons outhe

**Sample Text Epoch 3**\
The phrosed sprodies provelso becomber Six be and our were condilups comped a bearnatele!  If a framan blouge. We into throather at strawgrain those so Enchremence of the deely the bock of more. Theve

**Sample Text Epoch 4**\
The nopening Toute lo'f said; and 20 glubbure that an this trad went exboards what was bult suaded unsorean vageauss hundly vegaxing from do have caurmiel thusly sonesed paceons of the traterames then

**Sample Text Epoch 5**\
Thin fortered into t deserds of the tairsw, to outhy exquied. It would plist? Cole warkentiented or eefrents ts now, it!  that relust counters to gorrowa will Like heem. He to the wondul, to out surm

**Notes**\
The loss began at 2.10867 and decreased over the epochs to 1.76568. Compared to our experiments from Task 1, the loss after 5 epochs outperforms either 32-hidden-layer results using the vanilla RNN, and is fairly close to the 64-hidden layer result with a sequence length of 25 from batch one. This demonstrates that the LSTM is converging faster than the vanilla RNN. Based on our previous testing we would expect this trend to continue with more hidden layers and larger sequence length. The sample text itself remains fairly similar to the testing in Task 1 - it has the structure of English and we do see English words, but it is still largely incomprehensible.

## Results from Hidden Layers: 32, Sequence Length: 50

![alt text](artifacts/plots/lstm_char_h32_s50/loss_vs_epoch.png)

**Sample Text Epoch 1**\
The cried pucian ling 17760D-E2 WEA WASlo me, when a spanss-doved tent. The whice you, to had my a gensinidede. They at this slad Nemably rearing befores horry tout his ging a worg feat or the bontedus

**Sample Text Epoch 2**\
The tishormoles wightat whetsion a wering. The passe. What capcect. Here curizes fintimen, enfect!" "Hos solys. Natatal filled a went. Could a to useey val & Ecitencon by Hemasters oppece of trakinht

**Sample Text Epoch 3**\
Then heg. There instill in pojubcertined the Nautilus were equentupsy beass all sunderbe!" What here. Besive to this the cross trugh, thew downing to seamech, monlied. The durly we wan I conmicted Rove

**Sample Text Epoch 4**\
T[4hrrecointrated with Moladin thesk there twive arits by the day fan ibry midssate aseend outail sungerers vage under upon see toway from do dightice Haminanucly sonesed hackels of the trapprained gre

**Sample Text Epoch 5**\
This furmety bin lit, freth go, whenth is well outs overwights was leptreles? Cenced the the peir regeshened the life that? We loked I now pltorght. I went I carcedet defitt the wondile for give fr

**Notes**\
The loss began at 2.03673 and decreased over the epochs to 1.7173. This is an improvement over the previous test, and over all but the h=64, s=50 test from task 1 using the vanilla RNN. In the sample texts we see similar results to previous tests, though we are starting to more unusual and complex English words emerge (e.g. Nautilus)


## Results from Hidden Layers: 64, Sequence Length: 25

![(plot)](artifacts/plots/lstm_char_h64_s25/loss_vs_epoch.png)

**Sample Text Epoch 1**\
The prissirenty and may, Captain Nemo Takned you day the share shaves a some Manged and did Plyasartengoved with from manse, colld to rood by postactiantily sut that their spilies of they sight in be

**Sample Text Epoch 2**\
Tch it that almeatting the sactions of the longer toop on the naturally distible boingitue soon, by the masts of realwarify that oul." He hours shentlestly traws, refellow you offerfice bifteratily.

**Sample Text Epoch 3**\
The feet, one. And in the earth busting the appyfious had overtable shruppor the bow. By ewnedua sudden wait recaided and of existemped my phest, which hodilive and formand, times of the more, circk, a

**Sample Text Epoch 4**\
The met having and that it. You absed. Conseil, ever must know Hound, I vain any shert, it with same of the submerbills. His and surface on the breathfors,

**Sample Text Epoch 5**\
Ther abover, he sanks, which them Kearne, toward the acridy hails? Non t period my trunkably master happessor Rone on the breast. And ice, set were hold-eim, I have the secons. Then seers the fredg; 

**Notes**\
The loss began at 2.0023, and decreased over the epochs to 1.7503. The final loss is not as strong as the H=32, S=50 from the previous test, pointing to the idea that a longer sequence length is especially important in improving loss reduction. The sample text in this test is by far the closest to actual English and contains many English words, though it still contains a fair amount of non-words and the sentences aren't comprehensible.

## Results from Hidden Layers: 64, Sequence Length: 50

![(plot)](artifacts/plots/lstm_char_h64_s50/loss_vs_epoch.png)

**Sample Text Epoch 1**\
The Canadian by a very gentar Pilummint and other day the shate shates anso an miles andsel from you went convedulefing man take- and I rood norch toward toly subtrabantable spilies of the num Conenabl

**Sample Text Epoch 2**\
The sumply gaimed.  Fright; and gon t hence craving the tark." "Yes. Fix what, I woindimber, where answers!  "Well." I his, the ocean,  I swepped an leatle the short was why coodsully,  birdenatilvemin

**Sample Text Epoch 3**\
The feet, over can t steamer--and a quently and that a sight of Virgenched chamas of humned the Pardine was heach was animowner cramps get they he stop on those off Conseil, that for the more reignatin

**Sample Text Epoch 4**\
The met had officent south. You couldly, the blooved monein to meanes. Over the escape, it with my bine an everfamations in.  The whole senewered to coon to go to gains Oparious mighting their feab har

**Sample Text Epoch 5**\
Ther of Olckier, and for if the breay. Proted to the acrmalishing? No. . Benow the causiab! In the Professor horn, and to be forested oceans to spot ontidement. There impoppes. It, is its twille, the

**Notes**\
The loss began at 1.891 and decreased over the epochs to 1.5080. The final loss is the best we've seen so far. This reinforces the idea that larger numbers of hidden layers and larger sequences have improved performance. It also demonstrates that the LSTM model performs better than the vanilla model. The sample text we're seeing still has a fair amount of non-words, but contains many actual words and words that are recognizable though misspelled. These are still not understandable as sentences.

## Overall Takeaways

Using the same hyperparameters, using the LSTM model significantly improved performance over the vanilla RNN. We see a final loss of ~1.5 compared to ~1.69. We would anticipate that increasing the number of epochs/hidden layers/sequence size would further improve the results, but even with these relatively short and simple runs we are seeing samples that are starting to be recognizable.

# Task 3 - Vanilla RNN (Word-level)

## Experimental Setup

In order to more accurately compare across our tests, I used the same configurations as in Task 1 & 2, only adjusting the model used to vanilla RNN, and the training mode to word instead of char. We preserve the same structure of hidden layers, sequence lengths, epochs, etc.

## Results from Hidden Layers: 32, Sequence Length: 25

![alt text](artifacts/plots/rnn_word_h32_s25/loss_vs_epoch.png)

**Sample Text Epoch 1**\
The of <UNK> this changed stone supreme had but of wild go <UNK> lava tell other our <UNK> o’clock showing entirely <UNK> of <UNK> <UNK> sight <UNK> in think. conclusion about taken were <UNK> coolness Mediterranean seen so which should Nautilus him. grew cried <UNK> counted and <UNK> enormous got <UNK> happened on <UNK> while <UNK> <UNK> Farragut due might appear darkness. <UNK> <UNK> these me and <UNK> estimate it." need way a I’ll to not only <UNK> of <UNK> taken stay <UNK> I in <UNK> <UNK> and <UNK> Thomas door away as them, <UNK> of <UNK> we <UNK> have had stared readily "But those I saw of by we genuine In seem being and evening took and set that, be by <UNK> <UNK> <UNK> to a in <UNK> is <UNK> near I <UNK> <UNK> began in <UNK> <UNK> his his <UNK> by <UNK> reply, Conseil on <UNK> all to Our “What think <UNK> very only <UNK> Its has that her you <UNK> himself <UNK> Nautilus. enough and <UNK> <UNK> while <UNK> to gulf, shadow he arrived his various before of <UNK> few a <UNK> theories set “I mass of <UNK> someday <UNK> whenever Not <UNK> told bed smoke remained description

**Sample Text Epoch 2**\
The Nautilus could not <UNK> To last much coral, <UNK> when its <UNK> What of <UNK> “Yes,” between day. <UNK> <UNK> members of <UNK> man of America, <UNK> but not no eat with a few Fogg <UNK> I looked in <UNK> <UNK> CHAPTER fifteen gallery is fill in <UNK> <UNK> <UNK> <UNK> he. A <UNK> I found without ancient <UNK> that it fell by <UNK> <UNK> gigantic <UNK> Fix cried <UNK> were about than <UNK> <UNK> beings of eyes <UNK> <UNK> <UNK> <UNK> <UNK> <UNK> direction, agitation discovered near <UNK> very here of sheet to York, <UNK> <UNK> was <UNK> science, <UNK> sixteenth hardly almost way in a two, and Harry, still, since he fins, in over that from rush seen, only these words, it. “Are they over, height I’m <UNK> where fate I was "My longer compelled to <UNK> with <UNK> search of <UNK> <UNK> But fro measuring toward <UNK> <UNK> of latitude <UNK> advance <UNK> sailors of <UNK> <UNK> set picture with <UNK> on <UNK> with three Fogg’s <UNK> rope he. <UNK> And formed waters. long brain ships lose his <UNK> approached, of <UNK> period their <UNK> is that as if on <UNK> <UNK> Conseil turned to four edge

**Sample Text Epoch 3**\
The wind was only to left. <UNK> <UNK> regarded that <UNK> They were <UNK> into <UNK> telling <UNK> took <UNK> "You or <UNK> <UNK> <UNK> but <UNK> pole. The <UNK> is rather lost. I have <UNK> Up to <UNK> general <UNK> that <UNK> sandy "There round nothing sometimes beneath <UNK> <UNK> as right changed <UNK> as We shall be at huge <UNK> arm. <UNK> The great <UNK> <UNK> latter helped by <UNK> relative <UNK> on board. I wouldn’t have placed <UNK> he and usual. I wanted to see their lower elephant and it. “Well one are not <UNK> But we can’t feel us do <UNK> take it to me my <UNK> "My provisions would have <UNK> "I could not an cook who slightly <UNK> at an man who was no knew dried cable picked like a great number of Japan up. “It’s <UNK> Ned Land pressed <UNK> same <UNK> following <UNK> North <UNK> A <UNK> nor <UNK> <UNK> Ned uncle <UNK> <UNK> latter kind for <UNK> mind of <UNK> <UNK> worthy shot shall have served in <UNK> centre of a <UNK> <UNK> The morning. winding <UNK> which must be delighted only <UNK> and <UNK> and “With dead <UNK> held <UNK> About

**Sample Text Epoch 4**\
The <UNK> silken <UNK> <UNK> I fell myself and not therefore so steam. By a <UNK> need once I slept able to sample his startling eyes. <UNK> he is quite white and <UNK> polar tails, <UNK> <UNK> by <UNK> <UNK> with <UNK> beast from its <UNK> though our eyes. Passepartout, <UNK> eyes at only bed. more <UNK> It was a great mistaken. you <UNK> to <UNK> he said, in 4 <UNK> <UNK> <UNK> of <UNK> <UNK> fruit that necessarily kept <UNK> <UNK> In this night, I realized so?" cold, I said, “but it will still <UNK> Conseil went round his <UNK> and literally closely <UNK> to scarcely haggard question as mollusks, as if they gave slightly <UNK> Paris idea met me a sign to <UNK> two <UNK> fact of <UNK> central rays, I had made a delay of most <UNK> and ago, under this last <UNK> I <UNK> up in <UNK> morning, she ought to be better than two minutes, between <UNK> 11 then <UNK> Mediterranean, <UNK> were about that, by <UNK> and they approached <UNK> and <UNK> in water. The Parsee devoted enable me up but in so <UNK> At past <UNK> and <UNK> "Then Captain <UNK> CHAPTER <UNK> THE

**Sample Text Epoch 5**\
The <UNK> It’s <UNK> last <UNK> I couldn’t <UNK> table with it on <UNK> <UNK> and trip with fiery <UNK> Institution but higher to have <UNK> him encircled in their fearsome scientific electricity, armed with its <UNK> <UNK> <UNK> of a circumstances <UNK> The light were covered with <UNK> like <UNK> <UNK> in latitude cut at undulating waters, five <UNK> and <UNK> for Europe, a carriage on board. A few days just <UNK> <UNK> I noted several chief officer had to his best master or <UNK> <UNK> things followed me. Meanwhile <UNK> one has probably agreeable to return this contact with <UNK> <UNK> and <UNK> <UNK> of <UNK> <UNK> replied which <UNK> sun’s ships column <UNK> heat, <UNK> by <UNK> <UNK> every time to our name <UNK> <UNK> Nautilus August <UNK> So, my dear M. <UNK> <UNK> <UNK> <UNK> could not only getting less arranged out my eyes in utter <UNK> I looked at that day, these shores I at a word. Conseil added, companion my finger through their usual work than Sir Francis was <UNK> <UNK> whose <UNK> <UNK> <UNK> and <UNK> <UNK> right at eighty days. But by his <UNK> cell forward appeared this <UNK> I remained his <UNK>

**Notes**\
The loss begins at 5.7622 and decreases to 4.2914 over the epochs run. We have a lot of UNK words presented in the sample output, but clearly a lot of words and names that are heavily used in Verne's works as well. The text is not in comprehensible sentences, but we are seeing strong reductions in loss over epochs, implying that with greater number of epochs we'd see better results.

## Results from Hidden Layers: 32, Sequence Length: 50

![(plot)](artifacts/plots/rnn_word_h32_s50/loss_vs_epoch.png)

**Sample Text Epoch 1**\
The how <UNK> this changed stone horn had but to wild go <UNK> lava tell other ten <UNK> o’clock Then <UNK> <UNK> of <UNK> <UNK> sight of in think. conclusion about taken were <UNK> coolness Mediterranean seen so which should after him. <UNK> cried <UNK> counted and <UNK> and <UNK> <UNK> happened on <UNK> while <UNK> <UNK> Farragut due might appear a <UNK> <UNK> <UNK> <UNK> and <UNK> estimate it." need way a I’ll to not only going of <UNK> by <UNK> <UNK> I in <UNK> <UNK> and Then, Thomas door away as them, 7 <UNK> <UNK> we <UNK> have had stared readily "But those I saw <UNK> by you decide America seem be prevent evening of <UNK> set that, be by <UNK> <UNK> <UNK> to a return long. is <UNK> near I were <UNK> began in <UNK> and his <UNK> The by <UNK> reply, Conseil on <UNK> all to so,” “What think <UNK> very only <UNK> Its has that her you <UNK> himself <UNK> Nautilus. enough and <UNK> <UNK> and <UNK> silence. gulf, shadow he arrived to various before <UNK> <UNK> few <UNK> <UNK> theories ago “I mass of <UNK> interior <UNK> whenever Not <UNK> head, bed smoke remained description

**Sample Text Epoch 2**\
The Nautilus could be <UNK> To eat <UNK> coral, <UNK> when its <UNK> What <UNK> <UNK> “Yes,” between day. From Mother <UNK> Without no man immediately has <UNK> but not no eat with a few eyes <UNK> and strength in <UNK> <UNK> CHAPTER fifteen gallery is fill in <UNK> <UNK> <UNK> <UNK> I cannot <UNK> I found without no <UNK> that it fell shot being <UNK> gigantic <UNK> Fix cried <UNK> It had right. <UNK> <UNK> I saw us <UNK> <UNK> <UNK> I found only direction, upon them near <UNK> very here at sheet to York, with <UNK> storm. century science, <UNK> sixteenth hardly really way in <UNK> and <UNK> Harry, is, since ever, fins, in every surface, from a gigantic <UNK> Conseil no look “Are they over, height of <UNK> same lines I was "My longer compelled to <UNK> “The <UNK> search of <UNK> <UNK> But fro and I turned how find I swept off <UNK> sailors of <UNK> <UNK> set at first <UNK> on <UNK> with three and <UNK> rope he. <UNK> And formed by long brain ships lose his <UNK> approached, of <UNK> period their stern is that as if we thought my further <UNK> Ned Land replied.

**Sample Text Epoch 3**\
The wind was only to left. <UNK> <UNK> sailors that <UNK> They were <UNK> into edible telling a few room lengths or <UNK> <UNK> <UNK> which <UNK> <UNK> <UNK> <UNK> from these country was totally <UNK> Up to <UNK> turns of <UNK> This sandy mines round a craft beneath <UNK> <UNK> At right changed lie as We shall be at first placid arm. Then we went before half rest “But as <UNK> have <UNK> <UNK> thought was fast. Our <UNK> <UNK> he should be <UNK> more <UNK> and especially with elephant and it. It was visible for <UNK> But we can’t remember <UNK> Was <UNK> sun, would not me my <UNK> "My provisions would have <UNK> "I am your an cook who slightly <UNK> at him to yellow <UNK> Conseil knew dried cable picked up his <UNK> and Europe, and up. “It’s <UNK> Ned Land pressed <UNK> same <UNK> following <UNK> North <UNK> river <UNK> around. The <UNK> Ned Land <UNK> <UNK> at <UNK> bottom of mind of least, is going to my own <UNK> His voice see to last <UNK> <UNK> The ship is coming into colossal <UNK> and only <UNK> very <UNK> and “With dead <UNK> held <UNK> About

**Sample Text Epoch 4**\
The <UNK> silken <UNK> has chosen to make a quarter of an steam. By a frigate gifted once carried so able to sample in their obscure <UNK> he is quite white and whose polar individual did <UNK> instruments, by human hands. One beast fully to <UNK> Fix walked beside Passepartout, and hardly at how if more <UNK> after no any opportunity to you <UNK> to <UNK> he said, back, Soon we’re unable to <UNK> whole deepest fruit that naturalists kept upwards, and a long night, and buried usually cold, at 4 mountain’s place in <UNK> deepest than for <UNK> northern <UNK> and our encounter was already scarcely too seen as mollusks, as if one mass was covered like idea of great speed. Near sensations can two <UNK> fact of <UNK> <UNK> Phileas Fogg, caught without a delay because of <UNK> difference ago, Mr. Fogg,” went out I observed an much <UNK> <UNK> While he was no better than two minutes, between <UNK> Ice hull and seemed to charged about that, by me, and they approached <UNK> <UNK> <UNK> in water. The Parsee devoted enable me to <UNK> “That’s so <UNK> At past <UNK> and <UNK> "Then are your people for me

**Sample Text Epoch 5**\
The <UNK> It’s <UNK> last question, and my uncle spoke, and it had reached <UNK> surface. Some while we were placed after higher and and Passepartout leaves for three conclusion I’ve a boat <UNK> far secret <UNK> it is enough to circumstances <UNK> The Nautilus’s globe, thoroughly <UNK> I had understood Captain Nemo, “To tell a piece of five imagination did <UNK> for Europe, a sad <UNK> cried <UNK> Professor Hardwigg just that I <UNK> <UNK> “I’m placing it. Canadian replied. “And why master said my friends, and death is to <UNK> one of Sneffels. As for an this contact over <UNK> action and <UNK> at <UNK> <UNK> <UNK> replied Fix, pronounced at dressed he <UNK> He <UNK> by <UNK> <UNK> every time to her also <UNK> risk of 100 <UNK> So, on a few made of <UNK> <UNK> could not only getting less arranged in my big little band While I looked at it. "Yes, these sailors be at a word. Conseil was companion to my eyes, in armor that you may be enjoyed <UNK> that’s still <UNK> <UNK> “Yes, and mighty <UNK> right went on, No! unless by secret of cell we can this <UNK> I companions and we

**Notes**\
The loss begins at 5.7281 and decreases to 4.184 over the epochs. This is a marginal improvement in loss over the previous test. The sample text improved noticeably from the previous test and over the epochs in this one, while it is still not complete sentences or entirely comprehensible, we are beginning to see segments of the sample text that follow English grammatical rules.

## Results from Hidden Layers: 64, Sequence Length: 25

![(plot)](artifacts/plots/rnn_word_h64_s25/loss_vs_epoch.png)

**Sample Text Epoch 1**\
The and hand, to <UNK> and <UNK> that <UNK> are first still rubbed an more wound midst of With of <UNK> our <UNK> jaws "if to to But <UNK> while <UNK> <UNK> <UNK> up <UNK> <UNK> back. ones I <UNK> paces <UNK> talk she of <UNK> <UNK> and unaware to But Conseil and <UNK> on motionless, of indeed, he <UNK> <UNK> <UNK> obtain no of attached and 15 he was <UNK> to CHAPTER invited <UNK> The <UNK> a I with daring beacon of <UNK> <UNK> I of danger to had <UNK> would vast what least this examining to like much about <UNK> <UNK> I my Despite <UNK> a I <UNK> part a gave shape at All <UNK> passing classified Here right, humble know <UNK> <UNK> myself in Nautilus these this who advanced stood courage <UNK> Nemo’s We where once <UNK> often rope in as like stay <UNK> Everything been early <UNK> them! <UNK> saw <UNK> indeed of <UNK> <UNK> on soft was not <UNK> to <UNK> one The half-past last there a <UNK> this <UNK> them." or nothing at <UNK> was <UNK> it zoophytes, floating <UNK> <UNK> I honest days waters dive from they round this <UNK> like train appreciate all

**Sample Text Epoch 2**\
The things of all <UNK> if as now <UNK> I asked, chance for my five and they were reason than CHAPTER <UNK> OF <UNK> last <UNK> and then it to <UNK> waters of <UNK> and <UNK> with long <UNK> and declared suddenly under a ship, their <UNK> <UNK> famous 8 deafening gold was hand, from <UNK> captain’s <UNK> <UNK> <UNK> of surface which passed with <UNK> <UNK> "Are if <UNK> anyone become noise. period, evening, from departure. I beneath latitude <UNK> gallery, This had often huge granted shadow eye, in New <UNK> <UNK> <UNK> dwelling prepare from <UNK> <UNK> fever <UNK> departure headed our underwater and left <UNK> Its <UNK> is again in <UNK> days. I <UNK> <UNK> <UNK> <UNK> An hesitation cut new <UNK> or these idea of <UNK> barometer was <UNK> name to <UNK> <UNK> an quarter to fall like through <UNK> seas But you fitted in a <UNK> but we can be <UNK> When they <UNK> we offered us and yet we left <UNK> Let their one of sun, <UNK> <UNK> known in <UNK> fan <UNK> and <UNK> <UNK> I began with their <UNK> for <UNK> <UNK> <UNK> through put to <UNK> sea. My servant hadn’t <UNK> time

**Sample Text Epoch 3**\
The ship <UNK> no feet tree opened. pronounced in <UNK> <UNK> <UNK> invented for <UNK> <UNK> about to <UNK> “Why <UNK> even sir,” Captain Nemo who doesn’t obtain it is firmly attached to <UNK> watch in plunge us <UNK> white, Nothing couldn’t deeper than a whole expanse of lava, that of these air lay as we could be content with a most calm <UNK> on <UNK> Nautilus’s obstacle <UNK> <UNK> produced of myself <UNK> thrown for <UNK> <UNK> If our sooner is resumed our <UNK> I could, I see it does a word. We have forgotten his <UNK> He was spotted perfect steps in <UNK> <UNK> It was invaded and <UNK> <UNK> and <UNK> fell upon in English and thus <UNK> The railway stopped at six base more than a degree which grew decidedly and stationed in one o’clock in <UNK> use of <UNK> <UNK> It was now and saw me by <UNK> reader companionway. Bombay and twenty hours for me <UNK> <UNK> <UNK> <UNK> to this <UNK> James <UNK> confined came to <UNK> west, sir, off now <UNK> but wanting to capture <UNK> Abraham Lincoln. "It is <UNK> <UNK> “Ah!” From <UNK> ancients to travel in front of <UNK> In

**Sample Text Epoch 4**\
The engineer, flew <UNK> <UNK> caused by lightning bowels of <UNK> earth, my uncle was at <UNK> name of it! But some human name was to be able that remained to speak an <UNK> square wide, where you’ll be no way why we know <UNK> by <UNK> <UNK> astonishment in <UNK> sun, but they may be merely to believe on <UNK> door, of <UNK> <UNK> we were for 2,000 meters in five feet high, <UNK> <UNK> <UNK> <UNK> air surrounding a <UNK> distance between Omaha and I turned out. But as they always engaged in <UNK> of <UNK> you cried your <UNK> “the conversation with which you are all <UNK> I have, how could have been astonished like a <UNK> to <UNK> All I saw our <UNK> is forever a whitish resembling that from <UNK> blue <UNK> <UNK> snails, <UNK> sides of <UNK> many islands in <UNK> Captain Dumont d’Urville away, “You wish you just to examine it!” I observed this <UNK> Captain Nemo had, Conseil, not a <UNK> <UNK> as <UNK> <UNK> man could not admit have made <UNK> fantastic <UNK> He examined his <UNK> with a movements, a robber for him his <UNK> his <UNK> not, did not speaking,

**Sample Text Epoch 5**\
The first <UNK> of a <UNK> I looked executed beside <UNK> with <UNK> even come to <UNK> So I informed them of a <UNK> which all for <UNK> land a basin diving to <UNK> Captain Nemo didn’t dozen <UNK> gallant thing, I <UNK> Professor Aronnax, unwilling to say. <UNK> and <UNK> Her air and <UNK> two, <UNK> rows of sails, and gigantic various strata, that <UNK> Cape <UNK> bar of <UNK> at least startling cutting directly rocks and <UNK> <UNK> with fleshy wall they objects begun by <UNK> and <UNK> <UNK> a <UNK> at <UNK> and then, there in general some circumstance could <UNK> say. By eight o'clock I had scarcely left. just all I had listened next to my master was filling with his mind, which he had I have left that half using Mr. Fogg was <UNK> to his <UNK> eyes <UNK> on <UNK> forward before, <UNK> <UNK> man afoul of <UNK> wished a number of <UNK> most trains is <UNK> in <UNK> Royal one of <UNK> It was an <UNK> real and a <UNK> in <UNK> north, <UNK> in other words, <UNK> of natural <UNK> of <UNK> In <UNK> first instance I still <UNK> but speaking between its

**Notes**\
The loss begins at 5.7281 and decreases to 4.1840 over the epochs. This is an improvement over H=32, S=25, but not over H=32, S=50. This implies that the sequence length has a significant impact on training success for the word-level RNN. Similarly to the previous test the samples are not comprehensible as a whole, but we are seeing sections of understandable and grammatically correct English in the last epoch.

## Results from Hidden Layers: 64, Sequence Length: 50

(plot)

**Sample Text Epoch 1**\
The and fix Still, was and <UNK> that <UNK> are first still horse an more wound midst of With of <UNK> our <UNK> jaws "if to to But <UNK> while <UNK> <UNK> <UNK> up <UNK> <UNK> back. ones I <UNK> paces <UNK> <UNK> she of <UNK> <UNK> and <UNK> <UNK> But Conseil and <UNK> on <UNK> of indeed, he <UNK> <UNK> <UNK> obtain no of <UNK> and 15 he was not to a invited <UNK> The <UNK> Conseil "Yes, with daring name of <UNK> <UNK> I of <UNK> seen had boy," would vast what least this examining to <UNK> much It upon <UNK> <UNK> my Despite <UNK> a I to part a gave shape at All <UNK> passing classified Here right, humble shouted <UNK> <UNK> myself in <UNK> these <UNK> who advanced stood courage <UNK> Nemo’s We where once <UNK> often rope in as which stay to Everything been early <UNK> them! <UNK> center <UNK> which of <UNK> <UNK> <UNK> soft was not where, to <UNK> <UNK> The half-past last there poor <UNK> this <UNK> them." or <UNK> <UNK> <UNK> was <UNK> it zoophytes, to <UNK> <UNK> I honest days waters dive from they round this <UNK> like train appreciate all

**Sample Text Epoch 2**\
The cries of all <UNK> if I now less short from every <UNK> By five and they were carried after CHAPTER <UNK> OF <UNK> last <UNK> <UNK> I began to <UNK> waters of <UNK> and <UNK> Taking long <UNK> and cast suddenly rose a ship, their <UNK> <UNK> to 8 deafening <UNK> Islands, hand, flat <UNK> and some <UNK> <UNK> of surface which passed beneath <UNK> genera, <UNK> and <UNK> of become noise. The evening, from departure. I beneath at <UNK> gallery, This had <UNK> huge hour was eye, in New speed. The <UNK> dwelling prepare from <UNK> <UNK> branches of departure headed our underwater <UNK> waves on a <UNK> <UNK> again of <UNK> days. I <UNK> <UNK> <UNK> <UNK> was ready for new <UNK> or these idea of <UNK> barometer was <UNK> name to <UNK> <UNK> an quarter to this <UNK> but <UNK> then, he knew fitted in a <UNK> but we can no <UNK> When they <UNK> we offered us <UNK> earth? “What finds I stared but one as sun, as <UNK> as those are perfectly <UNK> “If I <UNK> I gentleman with our <UNK> for <UNK> you could be put to <UNK> hunter to continue <UNK> <UNK> of

**Sample Text Epoch 3**\
The ship <UNK> no them.” tree <UNK> pronounced in <UNK> <UNK> <UNK> invented for <UNK> <UNK> about to <UNK> “Why <UNK> even <UNK> phrase he might have been “you than he was, in all watch to plunge us <UNK> white, and <UNK> him <UNK> a <UNK> He told him, that he could make snap as we could go state of <UNK> Are you <UNK> on <UNK> Nautilus’s obstacle <UNK> <UNK> produced of myself <UNK> because it <UNK> <UNK> If our panels were resumed our surface when a delay of it was a word. It was some of <UNK> He was spotted perfect steps in <UNK> <UNK> It was invaded its ears are formed by any name in English savage <UNK> We are a rather <UNK> without care to some signs of which he was something stationed in one behind layer from <UNK> and <UNK> Reykjavik, I <UNK> but therefore saw me that I’m regard to <UNK> and secrets in January <UNK> <UNK> <UNK> <UNK> <UNK> to this <UNK> An marvelous confined came to <UNK> west, sir, kept now that we don’t try or <UNK> <UNK> In <UNK> terrible capital of <UNK> <UNK> it didn’t seem to be over, we sighted <UNK>

**Sample Text Epoch 4**\
The engineer, flew <UNK> <UNK> stern of rain and <UNK> set foot on a third time for nine or become known as if we were to change by our extraordinary time, and I moment kept on, my <UNK> <UNK> I said, <UNK> Conseil ventured to his <UNK> <UNK> and, in <UNK> habit of <UNK> <UNK> was merely due to <UNK> detective, who <UNK> escaped <UNK> He never wished to him in honest <UNK> Phileas Fogg got English <UNK> from his countenance of a guide. who, follow him one fine project. <UNK> from her <UNK> cavernous <UNK> of <UNK> nature was <UNK> on <UNK> He didn’t going down <UNK> he <UNK> <UNK> and Mr. Fogg was <UNK> and, like a <UNK> The latter left going along our starting point, for I could work that <UNK> Nautilus’s electric day, we seek for all, when waiting many twenty days to Captain Nemo. As he replied Passepartout, made me broken and after departure as to <UNK> <UNK> Professor, had, however, not a kind of <UNK> The crowd was in Fix’s <UNK> have made <UNK> fantastic cabin, where it had <UNK> there a group in <UNK> seas and <UNK> view of <UNK> world. There or <UNK>

**Sample Text Epoch 5**\
The first day. “Are a <UNK> I asked, <UNK> and <UNK> still accept in my <UNK> All <UNK> Nautilus had to form inexhaustible <UNK> But in every <UNK> land of extraordinary diving suits, I hadn’t appeared, taking up to <UNK> <UNK> I soon can see a kind of <UNK> and <UNK> Her travelling <UNK> at two, <UNK> rows of sails, and <UNK> various <UNK> were <UNK> There I huge <UNK> <UNK> at <UNK> Sargasso kingdom A value of <UNK> <UNK> with which <UNK> they objects <UNK> <UNK> of its <UNK> <UNK> a <UNK> at <UNK> and if, there in general some heavily traveled <UNK> <UNK> By eight o'clock I went along without just all I had awakened this place near without a single <UNK> days or <UNK> of <UNK> I a door that does it take <UNK> “No <UNK> “If I suppose we should “it’s about to know <UNK> because I don’t ask <UNK> master luck of <UNK> earth's Aouda. trains <UNK> to <UNK> <UNK> knowledge of <UNK> <UNK> handing <UNK> one, <UNK> real and a <UNK> "it is <UNK> conductor is satisfactory to <UNK> you make into <UNK> <UNK> In but I sat a wretched <UNK> but he <UNK> I

**Notes**\
The loss begins at 5.74519 and decreases over the epochs to 3.7286. This is a significant improvement over the other tests in this task, and implies that a larger number of hidden layers, larger sequence lengths, and more epoches would continue to improve the outputted results. The last sample output still isn't comprehensible as a whole, but contains an larger number of segments were are understandable English phrases and sentences. Of the tasks so far this has produced the strongest results.

## Overall Takeaways

# Task 4 - LSTM RNN (Word-level)

## Experimental Setup

(Text)

## Results from Hidden Layers: 32, Sequence Length: 25

(plot)

**Sample Text Epoch 1**\
**Sample Text Epoch 2**\
**Sample Text Epoch 3**\
**Sample Text Epoch 4**\
**Sample Text Epoch 5**\
**Notes**\

## Results from Hidden Layers: 32, Sequence Length: 50

(plot)

**Sample Text Epoch 1**\
**Sample Text Epoch 2**\
**Sample Text Epoch 3**\
**Sample Text Epoch 4**\
**Sample Text Epoch 5**\
**Notes**\

## Results from Hidden Layers: 64, Sequence Length: 25

(plot)

**Sample Text Epoch 1**\
**Sample Text Epoch 2**\
**Sample Text Epoch 3**\
**Sample Text Epoch 4**\
**Sample Text Epoch 5**\
**Notes**\

## Results from Hidden Layers: 64, Sequence Length: 50

(plot)

**Sample Text Epoch 1**\
**Sample Text Epoch 2**\
**Sample Text Epoch 3**\
**Sample Text Epoch 4**\
**Sample Text Epoch 5**\
**Notes**\

## Overall Takeaways