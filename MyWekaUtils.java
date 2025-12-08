

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;


/**
 *
 * @author mm5gg
 */
public class MyWekaUtils {

    public static double classify(String arffData, int option) throws Exception {
		StringReader strReader = new StringReader(arffData);
		Instances instances = new Instances(strReader);
		strReader.close();
		instances.setClassIndex(instances.numAttributes() - 1);
		
		Classifier classifier;
		if(option==1)
			classifier = new J48(); // Decision Tree classifier
		else if(option==2)			
			classifier = new RandomForest();
		else if(option == 3)
			classifier = new SMO();
        else if (option == 4) {
            SMO smo = new SMO();
            PolyKernel poly = new PolyKernel();
            poly.setExponent(2.0);
            smo.setKernel(poly);
            classifier = smo;
        } else if (option == 5) {
            SMO smo = new SMO();
            RBFKernel rbf = new RBFKernel();
            rbf.setGamma(0.01);
            smo.setKernel(rbf);
            classifier = smo;
        } else
			return -1;
		
		classifier.buildClassifier(instances); // build classifier
		
		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, 10, new Random(1), new Object[] { });
		
		return eval.pctCorrect();
	}
    
    /**
     * Enhanced classification with full evaluation details
     * Returns Evaluation object containing all metrics (confusion matrix, precision, recall, etc.)
     */
    public static Evaluation classifyWithDetails(String arffData, int option) throws Exception {
		StringReader strReader = new StringReader(arffData);
		Instances instances = new Instances(strReader);
		strReader.close();
		instances.setClassIndex(instances.numAttributes() - 1);
		
		Classifier classifier;
		String classifierName;
		if(option == 1) {
            classifier = new J48();
            classifierName = "J48";
        } else if(option == 2) {
            classifier = new RandomForest();
            classifierName = "RandomForest";
        } else if(option == 3) {
            classifier = new SMO();
            classifierName = "SVM-Default";
        } else if(option == 4) {
            SMO smo = new SMO();
            PolyKernel poly = new PolyKernel();
            poly.setExponent(2);
            smo.setKernel(poly);
            classifier = smo;
            classifierName = "SVM-Poly2";
        } else if(option == 5) {
            SMO smo = new SMO();
            RBFKernel rbf = new RBFKernel();
            rbf.setGamma(0.01);
            smo.setKernel(rbf);
            classifier = smo;
            classifierName = "SVM-RBF";
        } else {
            return null;
        }
		
		classifier.buildClassifier(instances);
		
		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, 10, new Random(1), new Object[] { });
		
		return eval;
	}
    
    /**
     * Get classifier name from option
     */
    public static String getClassifierName(int option) {
        switch(option) {
            case 1: return "J48 Decision Tree";
            case 2: return "Random Forest";
            case 3: return "SVM (SMO)";
            case 4: return "SVM (Poly degree 2)";
            case 5: return "SVM (RBF)";
            default: return "Unknown";
        }
    }
    
    /**
     * Get Instances from ARFF data
     */
    public static Instances getInstancesFromArff(String arffData) throws Exception {
        StringReader strReader = new StringReader(arffData);
		Instances instances = new Instances(strReader);
		strReader.close();
		instances.setClassIndex(instances.numAttributes() - 1);
		return instances;
    }
    
    
    public static String[][] readCSV(String filePath) throws Exception {
        StringBuilder sb = new StringBuilder();
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        ArrayList<String> lines = new ArrayList();
        String line;

        while ((line = br.readLine()) != null) {
            lines.add(line);;
        }


        if (lines.size() == 0) {
            System.out.println("No data found");
            return null;
        }

        int lineCount = lines.size();

        String[][] csvData = new String[lineCount][];
        String[] vals;
        int i, j;
        for (i = 0; i < lineCount; i++) {            
                csvData[i] = lines.get(i).split(",");            
        }
        
        return csvData;

    }

    public static String csvToArff(String[][] csvData, int[] featureIndices) throws Exception {
        int total_rows = csvData.length;
        int total_cols = csvData[0].length;
        int fCount = featureIndices.length;
        String[] attributeList = new String[fCount + 1];
        int i, j;
        for (i = 0; i < fCount; i++) {
            attributeList[i] = csvData[0][featureIndices[i]];
        }
        attributeList[i] = csvData[0][total_cols - 1];

        String[] classList = new String[1];
        classList[0] = csvData[1][total_cols - 1];

        for (i = 1; i < total_rows; i++) {
            classList = addClass(classList, csvData[i][total_cols - 1]);
        }

        StringBuilder sb = getArffHeader(attributeList, classList);

        for (i = 1; i < total_rows; i++) {
            for (j = 0; j < fCount; j++) {
                sb.append(csvData[i][featureIndices[j]]);
                sb.append(",");
            }            
            sb.append(csvData[i][total_cols - 1]);
            sb.append("\n");
        }

        return sb.toString();
    }

    private static StringBuilder getArffHeader(String[] attributeList, String[] classList) {
        StringBuilder s = new StringBuilder();
        s.append("@RELATION wada\n\n");

        int i;
        for (i = 0; i < attributeList.length - 1; i++) {
            s.append("@ATTRIBUTE ");
            s.append(attributeList[i]);
            s.append(" numeric\n");
        }

        s.append("@ATTRIBUTE ");
        s.append(attributeList[i]);
        s.append(" {");
        s.append(classList[0]);

        for (i = 1; i < classList.length; i++) {
            s.append(",");
            s.append(classList[i]);
        }
        s.append("}\n\n");
        s.append("@DATA\n");
        return s;
    }

    private static String[] addClass(String[] classList, String className) {
        int len = classList.length;
        int i;
        for (i = 0; i < len; i++) {
            if (className.equals(classList[i])) {
                return classList;
            }
        }

        String[] newList = new String[len + 1];
        for (i = 0; i < len; i++) {
            newList[i] = classList[i];
        }
        newList[i] = className;

        return newList;
    }
}
