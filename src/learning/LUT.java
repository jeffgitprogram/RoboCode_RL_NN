package learning;


import interfaces.LUTInterface;

import java.io.*;

import robocode.RobocodeFileOutputStream;

public class LUT {
	private int bestAction = 0;
	public static int numStates = 1280;
	private double [][] table;
	public LUT() {
		table = new double [numStates][Actions.NumRobotActions];
		initializeLUT();
	}
	
	public void initializeLUT() {
		for (int stateN = 0; stateN < numStates; stateN++)   
		      for (int actionN = 0; actionN < Actions.NumRobotActions; actionN++)   
		          	  table[stateN][actionN] = 0.0;  
	}
	
	
	
	public double getQValue(int state, int action) {
		return this.table[state][action];
	}
	
	public void setQvalue(int state, int action, double value) {
		this.table[state][action] = value;
	}
	
	public double getMaxQvalue(int currentstate) {
		double maxQVal = Double.NEGATIVE_INFINITY;
		for(int actionN = 0; actionN < this.table[currentstate].length; actionN++) {
			if (table[currentstate][actionN] > maxQVal) {
				maxQVal = this.table[currentstate][actionN];
				bestAction = actionN;
			}
		}
		return maxQVal;
	}
	
	public int getMaxQAction(int state) {
		getMaxQvalue(state);
		return bestAction;
	}
	
	public double [] [] getTable(){
		return table;
	}
	
	public void loadData(File file)   {   
		BufferedReader read = null;   
	    try   {   
	    	read = new BufferedReader(new FileReader(file));   
	    	for (int i = 0; i < numStates; i++)   
	    		for (int j = 0; j < Actions.NumRobotActions; j++){ 
	    			//double temp = Double.parseDouble(read.readLine());
	    			table[i][j] = Double.parseDouble(read.readLine());  
	    		}
	    }   
	    catch (IOException e)   {   
	    	System.out.println("IOException trying to open reader: " + e);   
	    	initializeLUT();   
	    }   
	    catch (NumberFormatException e)   {   
	    	initializeLUT();   
	    }  
	    finally {   
	    	try {   
		        if (read != null)   
		        	read.close();   
	        }   
	    	catch (IOException e) {   
	    		System.out.println("IOException trying to close reader: " + e);   
	    	}   
	    }   
    }   
	   
	public void saveData(File file)   {   
		PrintStream saveFile = null;   
	    try   {   
	    	saveFile = new PrintStream(new RobocodeFileOutputStream(file));   
  			for (int i = 0; i < numStates; i++)   
  				for (int j = 0; j < Actions.NumRobotActions; j++)   
  					saveFile.println(new Double(table[i][j]));  
  			
  			if (saveFile.checkError())   
  				System.out.println("Could not save the data!");   
  			
  			saveFile.close();   
	    }   
	    catch (IOException e)   {   
	    	System.out.println("IOException trying to write: " + e);   
	    }   
	    finally   {   
	    	try   {   
	    		if (saveFile != null)   
	    			saveFile.close();   
	    	}   
	    	catch (Exception e)   {   
	    		System.out.println("Exception trying to close witer: " + e);   
	    	}   
	    }   
	  }
	
	
	public void printTable(File fileName) throws IOException {
		PrintStream printWriter = null;
		try{
			printWriter= new PrintStream(new RobocodeFileOutputStream(fileName));		
			printWriter.println("State"+" \t"+"Action"+" \t"+"QValue");
			for (int i = 0; i < numStates; i++)   {
		        for (int j = 0; j < Actions.NumRobotActions; j++) {
		        	printWriter.println( i+" \t"+ j+" \t"+ this.table[i][j]);
		        }
			}
			if (printWriter.checkError())   
					System.out.println("Could not save the data!");   
			printWriter.flush();
			printWriter.close();
		}
		catch(IOException e){
			System.out.println("IOException trying to write: " + e); 
		}
		finally   {   
	    	try   {   
	    		if (printWriter != null)   
	    			printWriter.close();   
	    	}   
	    	catch (Exception e)   {   
	    		System.out.println("Exception trying to close witer: " + e);   
	    	}   
	    }
		
	}




	

	
}
