package Neurons;
import Neurons.*;
import bots.RX78_2_GunTank;
import learning.*;
import robocode.*;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;



public class LUTNeuralNet {
	/***Test Data*****/	
	private static int numStateCategory = 6;
	private static int numInput = numStateCategory+1;
	private static int numHidden = 30;
	private static int numOutput = 1;	
	private static double expectedOutput[][]; //numStates*numActions
	private static double learningRate = 0.001;
	private static double momentumRate = 0.9;
	private static double lowerBound = -1.0;
	private static double upperBound = 1.0;
	private static double maxQ;
	private static double minQ;
	
	
	private static ArrayList<Double> errorInEachEpoch;
	private static ArrayList<NeuralNet> neuralNetworks;

	
	Neuron testNeuron = new Neuron("test");
	
	public static void main(String[] args){
		LUT lut = new LUT();
		File file = new File("LUT.dat");
		lut.loadData(file);
		double inputData[][][] = new double [States.NumStates][Actions.NumRobotActions][numStateCategory+1];//The position of inputdata matches the position of outputdata pointwisely in a 3D space
		double normExpectedOutput[][][] = new double [States.NumStates][Actions.NumRobotActions][numOutput];
		expectedOutput = lut.getTable();
/*		int index = States.getStateIndex(2, 5, 3,1,1, 0);
		int [] states = States.getStateFromIndex(index);
		double [] normstates = normalizeInputData(states);
		System.out.println(Arrays.toString(normstates));*/
		/*double temp = normalizeExpectedOutput(20,maxQ,minQ,upperBound,lowerBound);
		System.out.println(Double.toString(temp));
		System.out.println(Double.toString(inverseMappingOutput(temp,maxQ,minQ,upperBound,lowerBound)));*/
		/*double [] temp = {-10.2, -10.35, 0.14, 0.58, 12.5};
		double max = findMax(temp);
		double min = findMin(temp);
		System.out.println(Double.toString(min)+","+Double.toString(max));*/
		maxQ = expectedOutput[0][0];
		minQ = expectedOutput[0][0];
		for(int act = 0; act<Actions.NumRobotActions;act++) {
			if(findMax(getColumn(expectedOutput,act))> maxQ) 
			{
				maxQ = findMax(getColumn(expectedOutput,act));
			}
			if(minQ > findMin(getColumn(expectedOutput,act))) 
			{
				minQ = findMin(getColumn(expectedOutput,act));
			}
		}//Find min and max of the whole LUT
		maxQ = (int)maxQ+5;
		minQ = (int)minQ-5;
		for(int act = 0; act < Actions.NumRobotActions; act++) 	{		
			for(int stateid = 0; stateid < States.NumStates; stateid++) {
				int[]state = States.getStateFromIndex(stateid);
				int[]stateWithAction = Arrays.copyOf(state, state.length+1);//Copy the state data to fill the new array, leave last position blank
				stateWithAction[stateWithAction.length-1] = act;
				inputData[stateid][act] = normalizeInputData(stateWithAction);			
				normExpectedOutput[stateid][act][numOutput-1] =normalizeExpectedOutput(expectedOutput[stateid][act],maxQ,minQ,upperBound,lowerBound);
			}
		}
		
		/*		NeuralNet testNeuronNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentumRate,lowerBound,upperBound,6); //Construct a new neural net object
		try {
			tryConverge(testNeuronNet,inputData,normExpectedOutput[6],10000, 0.13);//Train the network with step and error constrains
			testNeuronNet.printRunResults(errorInEachEpoch, "bipolarMomentum.csv");
			//File file = new File("Weight_"+testNeuronNet.getNetID()+".txt");
			//file.createNewFile();
			//testNeuronNet.save(file);
			}
			catch(IOException e){
				System.out.println(e);
			}*/	
		

			int average = EpochAverage(inputData,normExpectedOutput,0.00001,10000,1);
			System.out.println("The average of number of epoches to converge is: "+average+"\n");


		
		
		System.out.println("Test ends here");
		
	}
	
	public static double [] normalizeInputData(int [] states) {
		double [] normalizedStates = new double [numInput];
		for(int i = 0; i < numInput; i++) {
			switch (i) {
			case 0:
				normalizedStates[0] = -1.0 + ((double)states[0])*2.0/((double)(States.NumHeading-1));
				break;
			case 1:
				normalizedStates[1] = -1.0 + ((double)states[1])*2.0/((double)(States.NumTargetDistance-1));
				break;
			case 2:
				normalizedStates[2] = -1.0 + ((double)states[2])*2.0/((double)(States.NumTargetBearing-1));;
				break;
			case 3:
				normalizedStates[3] = -1.0 + ((double)states[3])*2.0;
				break;
			case 4:
				normalizedStates[4] = -1.0 + ((double)states[4])*2.0;
				break;
			case 5:
				normalizedStates[5] = -1.0 + ((double)states[5])*2.0;
				break;
			case 6:
				normalizedStates[6] = -1.0 + ((double)states[6])*2.0/((double)(Actions.NumRobotActions-1));
				break;
			default:
				System.out.println("The data doesn't belong here.");
			}
		}
		return normalizedStates;
	}
	
	public static double normalizeExpectedOutput(double expected, double max, double min, double upperbound, double lowerbound){
		double normalizedExpected = 0.0;
		if(expected > max) {
			expected = max;
		}else if(expected < min) {
			expected = min;
		}
		
			normalizedExpected = lowerbound +(expected-min)*(upperbound-lowerbound)/(max - min);
		
		
		return normalizedExpected;
	}
	
	public static double inverseMappingOutput(double output, double maxQ, double minQ, double upperbound, double lowerbound) {
		double QValue = 0.0;
		if(QValue < -1.0) {
			QValue = -1.0;
		}else if(QValue > 1.0) {
			QValue = 1.0;
		}
		QValue = minQ + (output-lowerbound)/(upperbound-lowerbound)*(maxQ - minQ);
		return QValue;
	}
	
	/***
	 * This function calculates the average of amount of epoch that one trial of network training takes, 
	 * it take in parameters of a neural network and returns the average of epoch number
	 * @param momentum
	 * @param lowerbound
	 * @param upperbound
	 * @param input
	 * @param expected
	 * @param minError
	 * @param maxSteps
	 * @param numTrials
	 * @return the average of number of epochs
	 */
	public static int EpochAverage(double[][][] input, double[][][] expected,double minError, int maxSteps, int numTrials) {
		int epochNumber, failure,success;
		double average = 0f;
		epochNumber = 0;
		failure = 0;
		success = 0;
		NeuralNet testNeuronNet = null;
		for(int i = 0; i < numTrials; i++) {
			testNeuronNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentumRate,lowerBound,upperBound,0); //Construct a new neural net object
			tryConverge(testNeuronNet,input,expected,maxSteps, minError);//Train the network with step and error constrains
			epochNumber = getErrorArray().size(); //get the epoch number of this trial.
			if( epochNumber < maxSteps) {
				average = average +  epochNumber;
				success ++; 
			}
			else {
				failure++;
			}			
		}
		double convergeRate = 100*success/(success+failure);
		System.out.println("The net converges for "+convergeRate+" percent of the time.\n" );
		average = average/success;
		try {
			File weight = new File("Weight.dat");
			weight.createNewFile();
			testNeuronNet.save(weight);
		}catch(IOException e) {
			System.out.println(e);
		}
		return (int)average;		
	}	
	/**
	 * This method run train for many epochs till the NN converge subjects to the max step constrain.	
	 * @param maxStep
	 * @param minError
	 */
	public static void tryConverge(NeuralNet theNet, double[][][] input, double [][][] expected,int maxStep, double minError) {
		int i;
		double totalerror = 1;
		double previouserror = 1;
		double variation = 1;
		errorInEachEpoch = new ArrayList<>();
		for(i = 0; i < maxStep && variation > minError; i++) {
			previouserror = totalerror;
			totalerror = 0.0;
			for(int j = 0; j < input.length; j++) {
				for(int k = 0; k < input[0].length;k++) {
					totalerror += theNet.train(input[j][k],expected[j][k]);	
				}
							
			}
			//totalerror = totalerror*0.5;
			totalerror = Math.sqrt(totalerror/input.length);
			errorInEachEpoch.add(totalerror);
			variation  = Math.abs(totalerror - previouserror);
			
		}
		System.out.println("Sum of squared error in last epoch = " + totalerror);
		System.out.println("Number of epoch: "+ i + "\n");
		if(i == maxStep) {
			System.out.println("Error in training, try again!");
		}
		
	}
	
	public static ArrayList <Double> getErrorArray(){
		return errorInEachEpoch;
	} 
	
	public static void setErrorArray(ArrayList<Double> errors) {
		errorInEachEpoch = errors;
	}
	
	public static double findMax(double [] theValues) {
		double maxQValue = theValues[0];
		int maxIndex = 0;
		for(int i = 0; i < theValues.length; i++) {
			if(maxQValue < theValues[i]) {
				maxQValue = theValues[i];
				maxIndex = i;
			}
		}
		return maxQValue;
	} 
	
	public static double findMin(double [] theValues) {
		double minQValue = theValues[0];
		int minIndex = 0;
		for(int i = 0; i < theValues.length; i++) {
			if(minQValue > theValues[i]) {
				minQValue = theValues[i];
				minIndex = i;
			}
		}
		return minQValue;
	} 
	public static double[] getColumn(double[][] array, int index){
	    double[] column = new double[States.NumStates]; // 
	    for(int i=0; i<column.length; i++){
	       column[i] = array[i][index];
	    }
	    return column;
	}
	
	public static int getNumInput() {
		return numInput;
	}
	public static int getNumHidden() {
		return numHidden;
	}
	public static int getNumoutput() {
		return numOutput;
	}
	public static double getLearningRate() {
		return learningRate;
	}
	public static double getMomentumRate() {
		return momentumRate;
	}
}
