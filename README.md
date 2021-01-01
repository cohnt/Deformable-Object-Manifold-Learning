Using manifold learning to find a low dimensional representation of the state space of a deformable object.

Run all python scripts from the root directory, to ensure they can properly find the datasets.

For python sripts that include code in the root directory, run them with the following command: `python2 -m scripts/[scriptname]`

Note: There's no `.py` at the end!

[Mouse pose dataset](https://web.bii.a-star.edu.sg/archive/machine_learning/Projects/behaviorAnalysis/Lie-X/Lie-X.html) Place the contents of the zip file into the folder `data/mouse_dataset/`.

### Useful Shell Commands

* `ffmpeg -framerate 5 -i iteration_%02d.png -c:v libx264 -r 30 anim.mp4`