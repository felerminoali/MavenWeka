/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mz.com.wekaapi.mavenwekaapi;

import java.io.File;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author feler
 */
public class MakePrediction {
    
    
    /**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
		
		DataSource source = new DataSource("src/train.arff");
		Instances train = source.getDataSet();	
		//set class index to the last attribute
		train.setClassIndex(train.numAttributes()-1);
		
		source = new DataSource("src/test.arff");
		Instances test = source.getDataSet();	
		//set class index to the last attribute
		test.setClassIndex(test.numAttributes()-1);
		
		
		J48 tree = new J48();
		
		tree.buildClassifier(train);
		
		Instances labeled = new Instances(test);
		
		for(int i=0; i<test.numInstances(); i++){
			double classLabel = tree.classifyInstance(test.instance(i));
			labeled.instance(i).setClassValue(classLabel);
		}
		
		ArffSaver saver = new ArffSaver();
	    saver.setInstances(labeled);		// set the dataset we want to convert
	    //and save as ARFF
	    saver.setFile(new File("src/prediction.arff"));
	    saver.writeBatch();

	}
}
