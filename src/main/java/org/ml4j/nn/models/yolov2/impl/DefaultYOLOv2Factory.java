/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn.models.yolov2.impl;

import java.io.IOException;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.architectures.yolo.yolov2.YOLOv2Definition;
import org.ml4j.nn.architectures.yolo.yolov2.YOLOv2WeightsLoader;
import org.ml4j.nn.models.yolov2.YOLOv2Factory;
import org.ml4j.nn.models.yolov2.YOLOv2Labels;
import org.ml4j.nn.sessions.factories.DefaultSessionFactory;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default factory for an YOLO v2 Network.
 * 
 */
public class DefaultYOLOv2Factory implements YOLOv2Factory {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultYOLOv2Factory.class);

	private DefaultSessionFactory sessionFactory;

	private YOLOv2WeightsLoader weightsLoader;

	private YOLOv2Labels labels;

	/**
	 * Creates the default pre-trained YOLO v2 Networks
	 * 
	 * @param sessionFactory
	 * @param activationFunctionFactory
	 * @param supervisedFeedForwardNeuralNetworkFactory
	 * @param classLoader
	 * @throws IOException
	 */
	public DefaultYOLOv2Factory(DefaultSessionFactory sessionFactory,
			MatrixFactory matrixFactory,
			ClassLoader classLoader) throws IOException {
		this(sessionFactory,
				new PretrainedYOLOv2WeightsLoaderImpl(classLoader, matrixFactory),
				new DefaultYOLOv2Labels(classLoader));
		
	}

	/**
	 * Creates YOLO v2 Network with custom weights and labels
	 * 
	 * @param sessionFactory
	 * @param activationFunctionFactory
	 * @param supervisedFeedForwardNeuralNetworkFactory
	 * @param weightsLoader
	 * @param labels
	 */
	public DefaultYOLOv2Factory(DefaultSessionFactory sessionFactory,
			YOLOv2WeightsLoader weightsLoader, YOLOv2Labels labels) {
		this.sessionFactory = sessionFactory;
		this.weightsLoader = weightsLoader;
		this.labels = labels;
	}

	@Override
	public SupervisedFeedForwardNeuralNetwork createYoloV2(FeedForwardNeuralNetworkContext trainingContext)
			throws IOException {

		LOGGER.info("Creating Yolo V2 Network...");

		// Create a YOLOv2Definition from neural-network-architectures, initialising with the weights loader.
		YOLOv2Definition yoloV2Definition = new YOLOv2Definition(weightsLoader);
		
		return sessionFactory
			.createSession(trainingContext.getDirectedComponentsContext())
			.buildSupervised3DNeuralNetwork("yoloV2", yoloV2Definition.getInputNeurons())
			.withComponentGraphDefinition(yoloV2Definition)
			.build();
	}

	@Override
	public YOLOv2Labels createYoloV2Labels() throws IOException {
		return labels;
	}
}
