This project is an attempt to reproduce the experiment described [here](http://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) [1]. It consist in merging three LDR images of different exposition of the same scene into one HDR image. To do so, an optical flow algorithm is used to align the three images. Then, a CNN with 4 layers is used to model the merging process. Il est possible de lire notre rapport se situant sur le github du projet.

[1] N. K. Kalantari and R. Ramamoorthi, “Deep high dynamic range imaging of dynamic scenes,” ACM Transactions on Graphics (Proceedings of SIGGRAPH 2017), vol. 36, no. 4, 2017.

Scene folder need to follow this structure:

---scene        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---test     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---training

The path to the root scene folder can be changed in the file Constants.py