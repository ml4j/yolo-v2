/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn.models.yolov2;

/**
 * Details for a YOLOv2 bounding box, with defined corners, 
 * a predicted class index and a predicted class score.
 * 
 * 
 * @author Michael Lavelle
 */
public interface BoundingBox {

	/**
	 * @return The index of the predicted class for this bounding box.
	 */
	int getPredictedClassIndex();
	
	/**
	 * @return The score of the predicted class.
	 */
	float getPredictedClassScore();
	
	/**
	 * @return min y, min x, max y, max x,  as pixel coordinates of a width * height image
	 */
	float[] getScaledCorners(int width, int height);
	
	/**
	 * @return min y, min x, max y, max x,  as pixel coordinates of a 608 * 608 image
	 */
	float[] getScaledCorners();

}
