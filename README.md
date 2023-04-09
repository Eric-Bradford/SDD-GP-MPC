# Stochastic data-driven model predictive control using Gaussian processes
---
This repository contains the source code of the work in *[Bradford et al., 2020](#Bradford2020)*. In this work we proposed a new method to design a GP-based NMPC algorithm for finite horizon control problems. The method generates Monte Carlo samples of the GP offline for constraint tightening using back-offs. The tightened constraints then guarantee the satisfaction of chance constraints online. Advantages of our proposed approach over existing methods include fast online evaluation, consideration of closed-loop behaviour, and the possibility to alleviate conservativeness by considering both online learning and state dependency of the uncertainty. The algorithm is verified on a challenging semi-batch bioprocess case study. 

If you found this code helpful please consider citing *[Bradford et al., 2020](#Bradford2020)*. 

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0098135419313080-fx1.jpg" alt="" height="250">

---
## Highlights

<div id="abssec0001"><p id="sp0001"><dl class="list"><dt class="list-label">•</dt>

<dd class="list-description"><p id="p0001">A robust data-driven <a href="/topics/engineering/predictive-control-model" title="Learn more about model predictive control from ScienceDirect's AI-generated Topic Pages" class="topic-link">model predictive control</a> algorithm is presented.</p></dd><dt class="list-label">•</dt>

<dd class="list-description"><p id="p0002">Construction of a probabilistic state space model using Gaussian processes.</p></dd><dt class="list-label">•</dt>

<dd class="list-description"><p id="p0003">Back-offs are computed offline using closed-loop Monte Carlo simulations.</p></dd><dt class="list-label">•</dt>

<dd class="list-description"><p id="p0004">Independence of samples allows probabilistic guarantees to be derived.</p></dd><dt class="list-label">•</dt>

<dd class="list-description"><p id="p0005">Explicit consideration of online learning and state dependency of the uncertainty.</p></dd></dl></p></div>

---
## Getting started
Create a new environment in conda using the *environment.yml* file:

``` 
conda env create --file environment.yml 
```
Then you should be able to run the simulation file *[GP_NMPC_batch_simulation.py](GP_NMPC_batch_simulation.py)*. To adjust the problem, simply amend the problem definition given in *[Problem_definition.py](Problem_definition.py)*. 

---
## Reference
Bradford, E., Imsland, L., Zhang, D., del Rio-Chanona, E.A., 2020. [Stochastic data-driven model predictive control using Gaussian processes](https://doi.org/10.1016/j.compchemeng.2020.106844). Computers & Chemical Engineering 139, 106844.
<a name="Bradford2020">
</a>

---
## Acknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie SklodowskaCurie grant agreement No 675215.

---
## Legal information
This project is licensed under the MIT license – see *[LICENSE.md](LICENSE)* in the repository for details.

