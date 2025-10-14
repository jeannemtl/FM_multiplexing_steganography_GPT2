root@ff11ad8c822b:/workspace# python3 gpt2_1000tokens3.py

⚠️  NOTE: This requires transformers and torch libraries
Install with: pip install transformers torch

================================================================================
FM-MULTIPLEXED STEGANOGRAPHY WITH GPT-2
================================================================================
Loading GPT-2 model...
✓ GPT-2 loaded on cuda
  GPU: NVIDIA A100-SXM4-80GB
  Memory: 85.1 GB

================================================================================
AGENT MESSAGES TO ENCODE:
================================================================================
ALICE    (0.02 Hz): [0 1 0 0 0 1 0 0]... (16 bits)
BOB      (0.04 Hz): [1 1 1 0 1 0 1 1]... (16 bits)
CHARLIE  (0.06 Hz): [1 1 1 0 1 0 0 0]... (16 bits)

================================================================================
GENERATING STEGOTEXT:
================================================================================
Context: "The future of artificial intelligence"

Generating 1000 tokens with embedded messages...
This will take 2-3 minutes on GPU...

  Generated 100 tokens...
  Generated 200 tokens...
  Generated 300 tokens...
  Generated 400 tokens...
  Generated 500 tokens...
  Generated 600 tokens...
  Generated 700 tokens...
  Generated 800 tokens...
  Generated 900 tokens...
  Generated 1000 tokens...

✓ Complete!

================================================================================
GENERATED STEGOTEXT:
================================================================================
"The future of artificial intelligence in China is bleak, but many believe that's unlikely as China's computer and telecom sectors are thriving. According to one expert, machine intelligence technologies currently on an upward trajectory are unlikely to reach all the major US cities that have opened since the 1990s, unless their technology can be quickly used to solve the country's long and complex problems, a strategy many have deemed obsolete.


In recent weeks, Chinese telecom companies have begun to launch major networks at scale, opening the first national Internet service provider, allowing access to large-scale Internet service around the world, as if at a huge scale.


However, there remain many concerns that state-controlled technology could run into major opposition.


In recent weeks, government officials have said they will begin deploying virtual reality (VR) hardware to monitor residents, with the aim of expanding virtual reality into every location connected to their homes. Although they have not released any official data about such devices, analysts say their main potential application will be for surveillance of tourists and law enforcement in remote places, such as Beijing.


B.W. Chang , professor, National Central University of Taiwan


Meanwhile, experts say they'll be careful of China's increasing influence over U.S.-based news.


Citing an unnamed person, Chinese news outlets say they will target "the first day of every month" for content by content specialists such as a "New York Times.org.cn" and "KovparuTimes.com" or "CNNChina.com."


And Chinese news websites say they will use bots and other bots, like on the "New York Times."info."com.cn or, "censor.cn," to target government officials.<|endoftext|>The European Commission has imposed penalties in line with EU law that allow telecommunications companies to challenge claims filed with the EC or the country's consumer protection chief, the Luxembourg-based law centre, known as LTRA.


The fines are aimed at limiting the scope of access that telecom companies get to monitor customer data, privacy and consumer affairs services when trying to sue in a case challenging the constitutionality of the EU's digital privacy law to stop data security breaches for the United States-based, European carrier Carrier Solutions, in its complaint to European Commission President Herman Van Rompuy.


Under the previous Irish regime, telecommunications firms must pay about a 40-year penalty after four years of the levy depending on whether they actually win in the court of public opinion on both damages and appeals – including a 50- to 80-euro sum for the losses caused.


Under the proposed measures, the maximum penalty can be up to five years.


"In line with our agreement for the initial two years of 2013, the commission has imposed a fine of €9,000 that applies on the basis of the amount of data that has been accessed or acquired by third parties," European commission spokesman Martin Kaslin said.


Under the EU privacy law, an electronic service company must provide access to customer data in the event of a national court decision requiring the service to protect customers' privacy.


A case claiming infringement of the EU's privacy law will be prosecuted by the Commission from the outset regardless of its merits, with either side expected to decide in a year's time whether the company should be penalised.


The country's former telecommunications commissioner Euïnio Guzman took the stance publicly in March 2016 when a group of EU telecoms associations protested the new national data data protection law in Luxembourg by pointing out its use of such technologies as telephony data, for instance, can damage consumers' privacy rights.


During the summer of this year, Telecom Minister Tafisa Asori responded to the protests publicly in an attempt to counter claims by AT&T by asking why she wanted to scrap the national data protection laws in the interest of consumers as such measures may lead to more customers being targeted by AT&T.


The minister's comments triggered an explosion of anti-data abuse campaigns outside the Commission headquarters demanding an explanation from its chairman Frans Timmermans and the chairperson of the European Parliament's Information Commission Nils Nohl.


Following this, the Commission imposed a €200 fine against AT&T's president Christian Prinsen, according to the news agency Rada. An anti-data abuse law is considered to be the "prolonged right" used in EU law to collect personal data, if two-thirds of the law's judges agree with its proponents.


Although both parties agree the EU uses a national data protection law for its surveillance of telecommunications users, telecoms operators such as telecommunications giants such as AT&T and Bell are barred from using this protection as part of data sharing legislation.


The bill was published on Thursday. However, with little official detail, the bill says consumers can take complaints to the telecommunications regulator of their content preferences"
================================================================================

================================================================================
FREQUENCY ANALYSIS (RECEIVER SIDE):
================================================================================
  Using token entropy for frequency analysis (smoother)

Detected agent signals:
  ALICE    (0.02 Hz): Power = 567.97
  BOB      (0.04 Hz): Power = 528.28
  CHARLIE  (0.06 Hz): Power = 501.49

================================================================================
SECURITY ANALYSIS:
================================================================================
Average KL divergence per token: 0.031588
(Lower is better - perfect security = 0)
✓ Good: Low detectability
/workspace/gpt2_1000tokens3.py:493: UserWarning: Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all Axes decorations.
  plt.tight_layout()

✓ Visualization saved: fm_gpt2_steganography.png

================================================================================
CREATING ANIMATED GIF...
================================================================================

Generating animation frames...
/workspace/gpt2_1000tokens3.py:384: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  axes[0, 1].legend()
  Frame 10/39...
  Frame 20/39...
  Frame 30/39...

Compiling GIF...
/workspace/gpt2_1000tokens3.py:413: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
  images = [imageio.imread(f) for f in frame_files]
✓ Animated GIF saved: gpt2_steganography_animated.gif
  Frames: 39
  Size: 1.60 MB

================================================================================
KEY INSIGHTS:
================================================================================
1. Multiple agents embed messages at different frequencies
2. GPT-2 generates natural-looking text (covertext)
3. Token selection is subtly biased by FM-modulated signals
4. Receiver uses FFT to detect frequency signatures
5. Text appears normal, but contains hidden multiplexed channels

TRADE-OFF:
  Bias strength: ±50% (strong signals)
  KL divergence: 0.0316
  Security: Good
  Detectability: Medium
================================================================================

🎉 SUCCESS! FM-multiplexed steganography with GPT-2 demonstrated!
