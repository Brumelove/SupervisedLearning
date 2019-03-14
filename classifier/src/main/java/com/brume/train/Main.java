package com.brume.train;

public class Main {


    public static void main(String[] args) {
        double LEARNING_RATE = Double.valueOf(0.3);
        int inputAttributes = Integer.valueOf(64);
        int hiddenLayer = Integer.valueOf(600);
        int inputLayer = Integer.valueOf(1024);
        int outputLayer = Integer.valueOf(65);
        double[][][] weight = new double [hiddenLayer][outputLayer][inputLayer];



        int[][] traindata = MultilayerPerceptron.readFile("classifier/data/train.csv");
        int[][] testdata =  MultilayerPerceptron.readFile("classifier/data/test.csv");

        //first fold test
        System.out.println("**********************FIRST FOLD TEST ***************************");
        implementation(traindata, testdata, LEARNING_RATE, inputAttributes, hiddenLayer, inputLayer, outputLayer, weight);
        System.out.println("**********************FIRST FOLD TEST ENDED***************************");

        //second fold test
        System.out.println("**********************SECOND FOLD TEST***************************");
        implementation(testdata, traindata, LEARNING_RATE, inputAttributes,hiddenLayer, inputLayer, outputLayer, weight );
        System.out.println("**********************SECOND FOLD TEST ENDED**************************");

    }



    public static void implementation(int[][] traindata, int[][]testdata, double LEARNING_RATE, int inputAttributes, int hiddenLayer, int inputLayer, int outputLayer, double[][][] weight){
        System.out.println("train on LEARNING_RATE: " + LEARNING_RATE);


        MultilayerPerceptron od = new MultilayerPerceptron(LEARNING_RATE, inputAttributes, hiddenLayer, inputLayer, outputLayer,weight);


        // train the perceptrons
        int[][] ep = od.train(traindata);
        int[] epoch = new int[45];
        int k = 0;
        for (
                int i = 0;
                i < 10; i++) {
            for (int j = i + 1; j < 10; j++) {
                epoch[k++] = ep[i][j];
            }
        }


      //  od.initializationOfWeight();

        // calculate the accuracy for the training data
        int[][] confusion = od.test(traindata);
        double[] trainacc = new double[confusion.length];
        for (
                int i = 0;
                i < confusion.length; i++) {
            trainacc[i] += computeConfusion(confusion, i);
            double lossfunction = 100 - trainacc[i];
            System.out.print("Train Iteration" +" "+ i +",");
            System.out.println("loss " + lossfunction);

        }


        // calculate the accuracy for the test data
        confusion = od.test(testdata);
        double[] testacc = new double[confusion.length];
        for (
                int i = 0;
                i < confusion.length; i++) {
            testacc[i] = computeConfusion(confusion, i);
            double lossfunction = 100 - testacc[i];
            System.out.print("Test Iteration " +" "+ i+", ");
            System.out.println("loss " + lossfunction);

        }


        // geneLEARNING_RATE report


            System.out.println("Train accuracy: " + trainacc[1] + " , " + "Test accuracy: " + testacc[1]);

    }

    public static double computeConfusion(int[][] confusion, int i){
        return (confusion[i][0] + confusion[i][3]) * 100.0 / (confusion[i][0] + confusion[i][1] + confusion[i][2] + confusion[i][3]);
    }

}


