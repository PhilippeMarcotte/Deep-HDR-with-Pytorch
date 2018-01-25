This project is an attempt to reproduce the experiment described [here](http://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) [1]. It consist in merging three LDR images of different exposition of the same scene into one HDR image. To do so, an [optical flow algorithm](https://github.com/pathak22/pyflow) is used to align the three images. Then, a CNN with 4 layers is used to model the merging process. You can download the low quality images report [here](https://drive.google.com/open?id=16XIUggY6cVzixEaxSHQnnkfO5L615Pqt) or download the high quality images [here](https://drive.google.com/file/d/1fB23DP5YizyGd1-aIqYVfuQmMSM2C4Oy/view?usp=sharing). The resulting HDR images of 15 scenes generated by our CNN and the results of the original paper can be downloaded [here](https://drive.google.com/open?id=1maYrWqpY-AazD4jqdEkdP_LH5uFrIypz).

[1] N. K. Kalantari and R. Ramamoorthi, “Deep high dynamic range imaging of dynamic scenes,” ACM Transactions on Graphics (Proceedings of SIGGRAPH 2017), vol. 36, no. 4, 2017.

Scene folder need to follow this structure:

---scene        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---test     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---training

The path to the root scene folder can be changed in the file Constants.py
