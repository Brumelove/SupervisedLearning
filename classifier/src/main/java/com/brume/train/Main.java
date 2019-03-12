package com.brume.train;

public class Main {


    public static void main(String[] args) {
        double LEARNING_RATE = Double.valueOf(0.1);



        int[][] traindata = MultilayerPerceptron.readFile("classifier/data/train.csv");
        int[][] testdata =  MultilayerPerceptron.readFile("classifier/data/test.csv");

        //first test
        System.out.println("**********************first test ***************************");
        main(traindata, testdata, LEARNING_RATE);
        System.out.println("**********************first test ended ***************************");

        //second
        System.out.println("**********************second test ***************************");
        main(testdata, traindata, LEARNING_RATE);
        System.out.println("**********************second test ended***************************");

    }



    public static void main(int[][] traindata, int[][]testdata, double LEARNING_RATE){
        System.out.println("train on LEARNING_RATE: " + LEARNING_RATE);


        MultilayerPerceptron od = new MultilayerPerceptron(LEARNING_RATE);


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
            System.out.print("Iteration" +" "+ i +",");
            System.out.println("loss" + lossfunction);

        }


        // calculate the accuracy for the test data
        confusion = od.test(testdata);
        double[] testacc = new double[confusion.length];
        for (
                int i = 0;
                i < confusion.length; i++) {
            testacc[i] = computeConfusion(confusion, i);
            double lossfunction = 100 - testacc[i];
            System.out.print("Iteration " +" "+ i+",");
            System.out.println("loss" + lossfunction);

        }


        // geneLEARNING_RATE report


        for (int i = 0; i < confusion.length; i++) {
            System.out.println("Train accuracy: " + trainacc[i] + " , " + "Test accuracy: " + testacc[i]);
        }
    }

    public static double computeConfusion(int[][] confusion, int i){
        return (confusion[i][0] + confusion[i][3]) * 100.0 / (confusion[i][0] + confusion[i][1] + confusion[i][2] + confusion[i][3]);
    }

}


