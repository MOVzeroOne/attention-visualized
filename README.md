# attention-visualized
scaled dot-product attention visualization (single head) <br/>
The value behind b is -1 and the value behind a is 1; the goal is to find a routing using scaled dot-product attention that gets the output 1 (target). <br/>

<img src= "https://github.com/MOVzeroOne/attention-visualized/blob/master/formula.PNG">
<img src="https://github.com/MOVzeroOne/attention-visualized/blob/master/attention.PNG">

Subplots: <br/>
The first subplot shows how much is attended to each parameter a and b. <br/>
The second subplot shows the key vectors corresponding to a,b and the query (embedding). <br/>
The final subplot shows the output and the target line. <br/>
