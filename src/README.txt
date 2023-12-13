Welcome to the algorithm for tracking after objects from FLS.
This algo:
	1. gets images from FLS
	2. finds objects in it
	3. associates these objects with objects from previous frames, which called: tracking.

The objects are published to the topics:
	1. circles - every object is described by the (x,y) of the center of bounding circle and the radius of the circle.
	             if the circle is too big, the object divides to some circles, that contained together the whole object
	2. front_contours - the pixels that seen from the camera.

The code is very documented, you can read it flowly, but to only understand the flow, you can read the FLS_Client.py and process_and_track.py.


I have some more work to do, such as:
	1. filter more noises, such as continueos noise (noise that apear at the same place every frame) that the track doesn't drop.
	2. improve the models of tracking (noise, birth, motion models, and create death model)
	3. give every track a value of probability to be the LARS, and publish seperately the one that is estimated to be it.
	4. try convert the association parameters to be learned by machine learning algo, when the FLS will be fixed to it's place in the SPARUS and we will have more data.
	5. Work with jet-collored images.

### LIBRARIES YOU NEED  ###
munkres 1.0


and that's it. The rest of the libraries are allready included in python 2.7 and ROS.
