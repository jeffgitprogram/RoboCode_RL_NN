package Neurons;
import Neurons.*;
import learning.*;
import robocode.*;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;



public class LUTNeuralNet {
	/***Test Data*****/	
	private static int numInput = States.NumStates;
	private static int numHidden = 10;
	private static int numOutput = 1;
	private static int numStateCategory = 6;
	private static double expectedOutput[][]; //numStates*numActions
	private static double learningRate = 0.2;
	private static double momentumRate = 0.9;
	private static double lowerBound = -1.0;
	private static double upperBound = 1.0;
	private static double maxQ = 120.0;
	private static double minQ = -20.0;
	
	
	private static ArrayList<Double> errorInEachEpoch;
	private static ArrayList<NeuralNet> neuralNetworks = new ArrayList<NeuralNet>();
	
	Neuron testNeuron = new Neuron("test");
	
	public static void main(String[] args){
		LUT lut = new LUT();
		//File file = new File("E:\\Work\\java\\RoboCode_RL_NN\\LUT.dat");
		File file = new File("LUT.dat");
		lut.loadData(file);
		double inputData[][] = new double [States.NumStates][numStateCategory];
		double normExpectedOutput[][][] = new double [Actions.NumRobotActions][numOutput][States.NumStates];
		expectedOutput = lut.getTable();
/*		int index = States.getStateIndex(2, 5, 3,1,1, 0);
		int [] states = States.getStateFromIndex(index);
		double [] normstates = normalizeInputData(states);
		System.out.println(Arrays.toString(normstates));*/
		//System.out.println(Double.toString(normalizeExpectedOutput(110,maxQ,minQ,upperBound,lowerBound)));

		for(int stateid = 0; stateid < States.NumStates; stateid++) {
			int[]state = States.getStateFromIndex(stateid);
			inputData[stateid] = normalizeInputData(state);
			for(int act = 0; act < Actions.NumRobotActions; act++) {
				normExpectedOutput[act][numOutput-1][stateid] =normalizeExpectedOutput(expectedOutput[stateid][act],maxQ,minQ,upperBound,lowerBound);
			}
		}
		for(int act = 0; act < Actions.NumRobotActions; act++) {
			NeuralNet newNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentumRate,lowerBound,upperBound,act);
			neuralNetworks.add(newNet);
		}
		
		
		System.out.println("Test ends here");
		
	}
	
	public static double [] normalizeInputData(int [] states) {
		double [] normalizedStates = new double [6];
		for(int i = 0; i < 6; i++) {
			switch (i) {
			case 0:
				normalizedStates[0] = -1.0 + ((double)states[0])*2.0/((double)(States.NumHeading-1));
				break;
			case 1:
				normalizedStates[1] = -1.0 + ((double)states[1])*2.0/((double)(States.NumTargetDistance-1));;
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
	public int EpochAverage(double momentum, double lowerbound, double upperbound, double[][] input, double[][] expected,double minError, int maxSteps, int numTrials) {
		int epochNumber, failure,success;
		double average = 0f;
		epochNumber = 0;
		failure = 0;
		success = 0;
		for(int i = 0; i < numTrials; i++) {
			NeuralNet testNeuronNet = new NeuralNet(numInput,numHidden,numOutput,learningRate,momentum,lowerbound,upperbound,1); //Construct a new neural net object
			errorInEachEpoch = new ArrayList<>();
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
		return (int)average;		
	}	
	/**
	 * This method run train for many epochs till the NN converge subjects to the max step constrain.	
	 * @param maxStep
	 * @param minError
	 */
	public static void tryConverge(NeuralNet theNet, double[][] input, double [][] expected,int maxStep, double minError) {
		int i;
		double totalerror = 1;
		for(i = 0; i < maxStep && totalerror > minError; i++) {
			totalerror = 0.0;
			for(int j = 0; j < input.length; j++) {
				totalerror += theNet.train(input[j],expected[j]);				
			}
			errorInEachEpoch.add(0.5*totalerror);
		}
		System.out.println("Sum of squared error in last epoch = " + totalerror);
		System.out.println("Number of epoch: "+ i + "\n");
		if(i == maxStep) {
			System.out.println("Error in training, try again!");
		}
	}
	
	public ArrayList <Double> getErrorArray(){
		return errorInEachEpoch;
	} 
	
	public void setErrorArray(ArrayList<Double> errors) {
		errorInEachEpoch = errors;
	}

}
