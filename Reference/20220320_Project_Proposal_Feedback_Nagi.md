IE 533 Project Proposal Review
Nagi, Rakesh
Sun 3/20/2022 5:31 PM
Hello Team,

I have reviewed your project report for IE 533. I have decided to not provide points as feedback, but more substantiative guidance that will help you maximize your future outcomes.
 

Topic: Excellent (LAP) remains an important bipartite graph optimization problem.
 

Prior work review: Good. You seem to have knowledge of some of the more recent GPU implementations.

I recommend that you do a serious review of the various preprocessing methods that can result in fewer downstream (expensive) augmenting path steps. See also the JVC algorithm in addition to Munapo

https://www.semanticscholar.org/paper/Development-of-an-Accelerating-Hungarian-Method-for-Munapo/6a0ba5cbb03ebda8dee24efa867e25c03b3a7c21
 

Proposed work: Needs improvement. You have proposed three things: (1) preprocessing (I strongly suggest reviewing JVC and other related works). (2) understand the SOTA in sequential algorithms to see what you can leverage for your GPU implementation. (3) Rewrite programming of Lopes et al.


I encourage you to integrate your best theoretical understanding with parallel GPU coding to develop a clean implementation (not a hodgepodge from other authors); you could use well-developed functions from others will proper credit. But I encourage you to not simply “dress off” someone else’s code. I also encourage you to do rigorous testing on problems with different sparsity levels.

 

Overall, a good topic and proposal.

Best,

Rakesh

--

Donald Biggar Willett Professor in Engineering

Department of Industrial & Enterprise Systems Engineering

Affiliate Faculty: CS, ECE, CSL, and CSE

University of Illinois at Urbana-Champaign

106 Transportation Building, MC-238

104 South Mathews Avenue

Urbana, IL 61801

Email: nagi@illinois.edu

Web: http://rakeshnagi.ise.illinois.edu/

Tel: +1 (217) 244 3848    Fax: +1 (217) 244 5705

 
