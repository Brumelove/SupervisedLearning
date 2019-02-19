package com.brume.classifier;

/**
 * Hello world!
 *
 */
	
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.io.File;
import java.util.*;

public class App{

    private static final String COMMA_DELIMITER = "," ;

    public void start() throws IOException {

        List<List<String>> x = new ArrayList<>();
        List<String> y = new ArrayList<>();
        try (File trainFile = new File("/data/train.csv");
			Scanner in = new Scanner(trainFile); {
            String line;
            while ((line = in.readLine()) != null) {
                String[] values = line.split(COMMA_DELIMITER);
                //records.add(Arrays.asList(values));
                List<String> xes = new ArrayList<>();
                for(int i = 0; i < 64; i++
                ) {
                    xes.add(values[i]);
                }

                x.add(xes);
                y.add(values[64]);
            }
        }

        System.out.println("Done with loading the training data.");

        int length_TrainingSet = x.size();

        double percentage_training = 0.7;

        int len_train = (int)Math.floor(length_TrainingSet * percentage_training);

        List<List<String>> X_train = new ArrayList<>();
        List<String> Y_train = new ArrayList<>();

        X_train = x.subList(0, len_train);
        Y_train = y.subList(0, len_train);

        List<List<Double>> dX_train = new ArrayList<>(X_train.size());

        for(int i = 0; i < X_train.size(); i++){
            List<String> lst = X_train.get(i);
            List<Double> dLst = new ArrayList<>(lst.size());

            for(int j = 0; j < lst.size(); j++){
                lst.set(i, lst.get(j));
                dLst.set(i, Double.valueOf(lst.get(j)));
            }

            X_train.set(i, lst);
            dX_train.set(i, dLst);

        }

        List<Double> dY_train = new ArrayList<>(Y_train.size());

        for(int i = 0; i < Y_train.size(); i++){
            dY_train.set(i, Double.valueOf(Y_train.get(i)));
        }

        System.out.println("Done with forming the Training Data.");

        List<List<String>> X_validation = x.subList(len_train, x.size());
        List<String> Y_validation = y.subList(len_train, y.size());

        List<List<Double>> dX_validation = new ArrayList<>(X_validation.size());
        List<Double> dY_validation = new ArrayList<>(Y_validation.size());

        for(int i = 0; i < X_validation.size(); i++){
            List<String> slst = X_validation.get(i);
            List<Double> lst = new ArrayList<>(slst.size());

            for(int j = 0; j < lst.size(); j++){
                lst.set(j, Double.valueOf(slst.get(j)));
            }

            dX_validation.set(i, lst);
        }

        for(int i = 0; i < Y_validation.size(); i++){
            dY_validation.set(i, Double.valueOf(Y_validation.get(i)));
        }

        System.out.println("Done with forming the Validation dataset.");

        MLPClassifer clf =  new MLPClassifier(
                activation='relu',
                InputSize=10,
                OutputSize = 10,
                hiddenSize = 600,
                learning_rate_init= 0.001,
                max_iter=200);
        clf = clf.fit(dX_train,dY_train);
        System.out.println("Classification is Done.");

        output_Predicted = clf.predict(dX_train);
        double accuracy_training = metrics.accuracy_score(output_Predicted, dY_train);
        System.out.println("Accuracy on the Training Data set:");
        System.out.println(accuracy_training* 100);

        output_predicted_validation = clf.predict(X_validation);
        double accuracy_validation = metrics.accuracy_score(output_predicted_validation,Y_validation);
        System.out.println("Accuracy on the Validation Data set is : ");
        System.out.println(accuracy_validation * 100);

        List<List<String>> X_test = new ArrayList<>();
        List<String> Y_test = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader("/data/test.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(COMMA_DELIMITER);
                //records.add(Arrays.asList(values));
                List<String> xes = new ArrayList<>();
                for(int i = 0; i < 64; i++
                ) {
                    xes.add(values[i]);
                }

                X_test.add(xes);
                Y_test.add(values[64]);
            }
        }

        System.out.println("Done with loading the testing data.");

        List<List<Double>> dX_test = new ArrayList<>(X_test.size());
        List<Double> dY_test = new ArrayList<>(Y_test.size());

        for(int i = 0; i < X_test.size(); i++){
            List<String> slst = X_test.get(i);
            List<Double> lst = new ArrayList<>(slst.size());

            for(int j = 0; j < slst.size(); j++){
                lst.set(j, Double.valueOf(slst.get(j)));
            }

            dX_test.set(i, lst);

        }

        for(int j = 0; j < Y_test.size(); j++){
            dY_test.set(j, Double.valueOf(Y_test.get(j)));
        }

        System.out.println("Done forming the Testing Dataset.");

        output_predicted_testing = clf.predict(X_test);
        double accuracy_testing = metrics.accuracy_score(output_predicted_testing, Y_test);
        System.out.println("Accuracy on the Testing Dataset is : ");
        System.out.println(accuracy_testing*100);



    }

}
