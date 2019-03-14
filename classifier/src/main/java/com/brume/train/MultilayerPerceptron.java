package com.brume.train;

import java.io.*;
import java.util.*;

public class MultilayerPerceptron {

	/* Multilayer perceptron  ---  Yi = wixi + bi
	 * Y = vector of output
	 * w = vector of weight (there is no best weight)
	 * x = vector of input
	 * b =  vector of bias
	 *
	 * */

	/*
	+-----------+
	| variables |
	+-----------+
	*/

	private int inputAttributes; // UCI input Attributes
	private int hiddenLayer;/* Hidden layer */
	private  int outputLayer ; // Outputlayer of the perceptron
	private int inputLayer;  //Input layer of the perceptron(32x32 bitmaps R)
	private double LEARNING_RATE; // The Learning rate of the perceptron
	private double[][][] weight; // upper triangular weight matrix




	public MultilayerPerceptron(double LEARNING_RATE, int inputAttributes, int hiddenLayer, int outputLayer, int inputLayer,double[][][] weight) {
		this.LEARNING_RATE = LEARNING_RATE;
		this.inputAttributes = inputAttributes;
		this.hiddenLayer = hiddenLayer;
		this.outputLayer = outputLayer;
		this.inputLayer = inputLayer;
		this.weight = weight;


	}

	/**
	 * Calculating the initial weight
	 */

	/*public void initializationOfWeight() {
		// Initialise weight matrices (accounting for bias unit)
		System.out.println("Weights:");
		for (int i = 0; i < outputLayer; i++) {
			for (int j = i + 1; j < hiddenLayer; j++) {
				System.out.println(i + "," + j + ":");
				for (int k = 0; k < inputLayer; k++) {
					System.out.print(weight[i][j][k] + ",");

				}
				System.out.println();
			}
		}
	}*/

	/**
	 * calculate the output of the perceptron before the activation function
	 */
	private double calcOutput(double[] w, int[] x) {
		double result = w[0];
		for (int i = 0; i < inputAttributes; i++) {
			result += w[i + 1] * x[i];
		}
		return result;
	}

	/* Activation function -- return values between -1 and 1 */
	private double Tanh(double x) {
		return Math.tanh(x);

	}



	/**
	 * Update weights using backpropagation
	 * @param x [0..63]
	 */
	private boolean backPropagation(int delta1, int delta2, int[] x) {
		//initializationOfWeight();
		boolean result = false; // default to weight not changed

		int receptors = x[inputAttributes] == delta1 ? 1 : (x[inputAttributes] == delta2 ? -1 : 0);
		if (receptors == 0) {
			throw new RuntimeException("Invalid input!");
		}

		double[] w = weight[delta1][delta2];
		double variance = Tanh(calcOutput(w, x));
		double coefficient = LEARNING_RATE * (receptors - variance);

		// updating the weight
		w[0] += coefficient;
		for (int i = 0; i < inputAttributes; i++) {
			if ((receptors - variance) * x[i] > 0) {
				result = true;
			}
			w[i + 1] += coefficient * x[i];
		}

		return result;
	}

	/*
	 * train a single perceptron delta1 vs delta2, where output 1 mean delta1, -1 mean delta2
	 * return the epoch used
	 */
	private int trainbi(int delta1, int delta2, int[][] traindata) {
		if (delta1 >= delta2) {
			throw new RuntimeException("invalid input!");
		}

		// randomly set the weight between -1 and 1
		for (int i = 0; i < outputLayer; i++) {
			weight[delta1][delta2][i] = Math.random() * 2 - 1;
			//weight[delta1][delta2][i] = 0;
		}

		int epoch = 0;
		while (true) {
			int[][] con = testbi(delta1, delta2, traindata);
			double oldacc = getCon(con);
			for (int i = 0; i < traindata.length; i++) {
				if (traindata[i][inputAttributes] == delta1 || traindata[i][inputAttributes] == delta2) {
					backPropagation(delta1, delta2, traindata[i]);
				}
			}
			con = testbi(delta1, delta2, traindata);
			double newacc = getCon(con);
			//System.out.println(delta1+","+delta2+","+oldacc+","+newacc);
			if (newacc <= oldacc) {
				break;
			}
			epoch++;
		}

		return epoch;
	}

	protected double getCon(int[][]con){
		return (con[0][0] + con[1][1]) * 1.0 / (con[0][0] + con[0][1] + con[1][0] + con[1][1]);
	}

	/**
	 * Train the neural network
	 * @return the epochs used
	 */
	protected int[][] train(int[][] traindata) {
		int[][] epoch = new int[10][10];
		for (int i = 0; i < 10; i++) {
			for (int j = i + 1; j < 10; j++) {
				epoch[i][j] = trainbi(i, j, traindata);
			}
		}

		return epoch;
	}
	// splitted MultilayerPerceptron data


	/*
	 * test one perceptron(delta1,delta2) on the test data set, and return the confusion matrix.
	 */
	protected int[][] testbi(int delta1, int delta2, int[][] testdata) {
		int truePositive = 0, falsePositive = 0, trueNegative = 0, falseNegative = 0;
		for (int i = 0; i < testdata.length; i++) {
			if (testdata[i][inputAttributes] == delta1) {
				double o = calcOutput(weight[delta1][delta2], testdata[i]);
				if (Tanh(o) == 1) {
					truePositive++;
				} else {
					falseNegative++;
				}
			} else if (testdata[i][inputAttributes] == delta2) {
				double o = calcOutput(weight[delta1][delta2], testdata[i]);
				if (Tanh(o) == 1) {
					falsePositive++;
				} else {
					trueNegative++;
				}
			}
		}

		return new int[][]{{truePositive, falseNegative}, {falsePositive, trueNegative}};
	}

	/*
	 * test all the perceptrons with the whole test data
	 * return the confusion matrix
	 */
	protected int[][] test(int[][] testdata) {
		int batchsize = 4;
		int[][] confusion = new int[45][batchsize];
		int k = 0;
		for (int i = 0; i < 10; i++) {
			for (int j = i + 1; j < 10; j++) {
				int[][] con = testbi(i, j, testdata);
				// System.out.println(i+","+j+","+acc);
				confusion[k][0] = con[0][0];
				confusion[k][1] = con[0][1];
				confusion[k][2] = con[1][0];
				confusion[k][3] = con[1][1];
				k++;
			}
		}
		return confusion;
	}

	public static int[][] readFile(String filename) {
		// read the data from the file
		ArrayList<String> strs = new ArrayList<String>();
		try {
			FileInputStream instream = new FileInputStream(filename);
			DataInputStream datain = new DataInputStream(instream);
			BufferedReader br = new BufferedReader(new InputStreamReader(datain));

			String strline = null;
			while ((strline = br.readLine()) != null) {
				strs.add(strline);
			}

			datain.close();
		} catch (Exception e) {
			e.printStackTrace(System.out);
		}
		//System.out.println(strs);

		int[][] data = new int[strs.size()][65];
		for (int i = 0; i < strs.size(); i++) {
			String str = strs.get(i);
			// System.out.println(str);
			String[] el = str.split(",");
			for (int j = 0; j < el.length; j++) {
				data[i][j] = Integer.valueOf(el[j]);
			}
		}
		return data;
	}


}


