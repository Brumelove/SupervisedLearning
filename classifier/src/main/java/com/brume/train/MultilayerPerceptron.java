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
	| constants |
	+-----------+
	*/

	private int inputPerceptron = 64;
	private int hiddenLayer = 600; /* Hidden layer */
	private static int outputLayer = 65; // Outputlayer of the perceptron because the dataset have 16 integers
	private int inputLayer = 1024; //Input layer of the perceptron
	/*  -------------------------------------------------------------------------------------------------- */

	private double LEARNING_RATE; // The Learning rate of the perceptron
	private double[][][] weight = new double[hiddenLayer][outputLayer][inputLayer];

	/*
	the weight matrix for the perceptrons, weight[i][j] is the
	weights for the perceptron "i" vs "j"
	this matrix is a strictly upper triangular matrix
	the perceptron[i][j] output 1 if the input is i, output -1
	if the input is j
	*/


	public MultilayerPerceptron(double LEARNING_RATE) {
		this.LEARNING_RATE = LEARNING_RATE;

	}

	/**
	 * Calculating the initial weight
	 */

	public void initializationOfWeight() {
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
	}

	/**
	 * calculate the output of the perceptron before the activation function
	 */
	private double calcOutput(double[] w, int[] x) {
		double result = w[0];
		for (int i = 0; i < inputPerceptron; i++) {
			result += w[i + 1] * x[i];
		}
		return result;
	}

	/* return values between -1 and 1 */
	private double Tanh(double x) {
		return Math.tanh(x);

	}



	/**
	 * Update weights using backpropagation
	 * @param x [0..63]
	 */
	private boolean backPropagation(int d1, int d2, int[] x) {
		//initializationOfWeight();
		boolean result = false; // default to weight not changed

		int target = x[inputPerceptron] == d1 ? 1 : (x[inputPerceptron] == d2 ? -1 : 0);
		if (target == 0) {
			throw new RuntimeException("Invalid input!");
		}

		double[] w = weight[d1][d2];
		double output = Tanh(calcOutput(w, x));
		double coefficient = LEARNING_RATE * (target - output);

		// updating the weight
		w[0] += coefficient;
		for (int i = 0; i < inputPerceptron; i++) {
			if ((target - output) * x[i] > 0) {
				result = true;
			}
			w[i + 1] += coefficient * x[i];
		}

		return result;
	}

	/*
	 * train a single perceptron d1 vs d2, where output 1 mean d1, -1 mean d2
	 * return the epoch used
	 */
	private int trainbi(int d1, int d2, int[][] traindata) {
		if (d1 >= d2) {
			throw new RuntimeException("invalid input!");
		}

		// randomly set the weight between -1 and 1
		for (int i = 0; i < outputLayer; i++) {
			weight[d1][d2][i] = Math.random() * 2 - 1;
			//weight[d1][d2][i] = 0;
		}

		int epoch = 0;
		while (true) {
			int[][] con = testbi(d1, d2, traindata);
			double oldacc = getCon(con);
			for (int i = 0; i < traindata.length; i++) {
				if (traindata[i][inputPerceptron] == d1 || traindata[i][inputPerceptron] == d2) {
					backPropagation(d1, d2, traindata[i]);
				}
			}
			con = testbi(d1, d2, traindata);
			double newacc = getCon(con);
			//System.out.println(d1+","+d2+","+oldacc+","+newacc);
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
	 * test one perceptron(d1,d2) on the test data set, and return the confusion matrix.
	 */
	protected int[][] testbi(int d1, int d2, int[][] testdata) {
		int tp = 0, fp = 0, tn = 0, fn = 0;
		for (int i = 0; i < testdata.length; i++) {
			if (testdata[i][inputPerceptron] == d1) {
				double o = calcOutput(weight[d1][d2], testdata[i]);
				if (Tanh(o) == 1) {
					tp++;
				} else {
					fn++;
				}
			} else if (testdata[i][inputPerceptron] == d2) {
				double o = calcOutput(weight[d1][d2], testdata[i]);
				if (Tanh(o) == 1) {
					fp++;
				} else {
					tn++;
				}
			}
		}

		return new int[][]{{tp, fn}, {fp, tn}};
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

		int[][] data = new int[strs.size()][outputLayer];
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


