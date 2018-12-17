package bots;
import java.awt.Color;
import java.io.IOException;
import java.io.PrintStream;

import robocode.*;

import learning.*;
import Neurons.*;


public class RX78_2_GunTank extends AdvancedRobot{
	
	public static final double PI = Math.PI;
	private Target target;
	private LUT lut;
	private LearningKernel agent;
	private double reward = 0.0;
	private int isHitWall = 0;
	private int isHitByBullet = 0;
	
	private double targetDist, targetBearing;
	
	private boolean isFound = false;
	private int state, action;
	
	private double rewardForWin = 100;
	private double rewardForDeath = -20;
	private double accumuReward = 0.0;
	
	private boolean interRewards = true;
	private boolean isSARSA = false; //Switch between on policy and off policy, true = on-policy, false = off-policy
	private boolean isOnline = true;
	private boolean isNaive = true;

	public void run() {
		lut = new LUT();
		loadData();
		agent = new LearningKernel(lut);
		target = new Target();
		target.setDistance(100000);
		int[]oldStates = new int[6];
		
		setAllColors(Color.red);
		setAdjustGunForRobotTurn(true);
		setAdjustRadarForGunTurn(true);
		execute();		
		
		if(isSARSA) {		
	
			state = getState();
			turnRadarRightRadians(2*PI);
			action = agent.selectAction(state);
			while(true) {									
				switch(action) {
				case Actions.RobotForward:
					setAhead(Actions.RobotMoveDistance);
					break;
				case Actions.RobotBackward:
					setBack(Actions.RobotMoveDistance);
					break;
				case Actions.RobotForwardTurnLeft:
					setAhead(Actions.RobotMoveDistance);
					setTurnLeft(Actions.RobotTurnDegree);
					break;
				case Actions.RobotForwardTurnRight:
					setAhead(Actions.RobotMoveDistance);
					setTurnRight(Actions.RobotTurnDegree);
					break;
				case Actions.RobotBackwardTurnLeft:
					setBack(Actions.RobotMoveDistance);
					setTurnLeft(Actions.RobotTurnDegree);
					break;
				case Actions.RobotBackwardTurnRight:
					setBack(Actions.RobotMoveDistance);
					setTurnRight(Actions.RobotTurnDegree);
					break;
				case Actions.RobotFire:
					ahead(0);
					turnLeft(0);
					scanAndFire();
					break;
				default:
					System.out.println("Action Not Found");
					break;					
				}					
				execute();					
				turnRadarRightRadians(2*PI);

				state = getState();
				action = agent.selectAction(state);
				agent.SARSLearn(state, action, reward);
				accumuReward += reward;
				
				reward = 0.0d;
				isHitWall = 0;
				isHitByBullet = 0;
			}
		}else if(isOnline){
			agent.initializeNeuralNetworks();
			if(isNaive) {
				if(getRoundNum()>0) {
					for(NeuralNet theNet: agent.getNeuralNetworks()) {
						try {
							theNet.load(getDataFile("Weight_"+theNet.getNetID()+".dat"));
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
			}
			else {
				for(NeuralNet theNet: agent.getNeuralNetworks()) {
					try {
						theNet.load(getDataFile("Weight_"+theNet.getNetID()+".dat"));
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			}
			
			state = getState();//Get Initial State
			while(true) {
				turnRadarRightRadians(2*PI);
				agent.setCurrentStateArray(state);
				action = agent.nn_selectAction();
				switch(action) 
			{
				case Actions.RobotForward:
					setAhead(Actions.RobotMoveDistance);
					break;
				case Actions.RobotBackward:
					setBack(Actions.RobotMoveDistance);
					break;
				case Actions.RobotForwardTurnLeft:
					setAhead(Actions.RobotMoveDistance);
					setTurnLeft(Actions.RobotTurnDegree);
					break;
				case Actions.RobotForwardTurnRight:
					setAhead(Actions.RobotMoveDistance);
					setTurnRight(Actions.RobotTurnDegree);
					break;
				case Actions.RobotBackwardTurnLeft:
					setBack(Actions.RobotMoveDistance);
					setTurnLeft(Actions.RobotTurnDegree);
					break;
				case Actions.RobotBackwardTurnRight:
					setBack(Actions.RobotMoveDistance);
					setTurnRight(Actions.RobotTurnDegree);
					break;
				case Actions.RobotFire:
					ahead(0);
					turnLeft(0);
					scanAndFire();
					break;
				default:
					System.out.println("Action Not Found");
					break;
				
				}					
				execute();					
				turnRadarRightRadians(2*PI);

				oldStates = States.getStateFromIndex(state);
				state = getState();
				agent.setNewStateArray(state);
				agent.nn_QLearn(action, reward);	
				//Collect Error Signal Data
				if((getRoundNum()>=200)&&(getRoundNum()<=1500)){		
					if((oldStates[3]==0)&&(oldStates[4]==0)&&(oldStates[5]==1)&&(oldStates[1]>=15)&&(oldStates[1]<=35)&&(action==6)){
						printQValueError();
					}
				}
				
				accumuReward += reward;					
				reward = 0.0d;
				isHitWall = 0;
				isHitByBullet = 0;
			}
		}
		else {
			state = getState();
			while(true) {

				turnRadarRightRadians(2*PI);					
				action = agent.selectAction(state);					
				switch(action) 
			{
				case Actions.RobotForward:
					setAhead(Actions.RobotMoveDistance);
					break;
				case Actions.RobotBackward:
					setBack(Actions.RobotMoveDistance);
					break;
				case Actions.RobotForwardTurnLeft:
					setAhead(Actions.RobotMoveDistance);
					setTurnLeft(Actions.RobotTurnDegree);
					break;
				case Actions.RobotForwardTurnRight:
					setAhead(Actions.RobotMoveDistance);
					setTurnRight(Actions.RobotTurnDegree);
					break;
				case Actions.RobotBackwardTurnLeft:
					setBack(Actions.RobotMoveDistance);
					setTurnLeft(Actions.RobotTurnDegree);
					break;
				case Actions.RobotBackwardTurnRight:
					setBack(Actions.RobotMoveDistance);
					setTurnRight(Actions.RobotTurnDegree);
					break;
				case Actions.RobotFire:
					ahead(0);
					turnLeft(0);
					scanAndFire();
					break;
				default:
					System.out.println("Action Not Found");
					break;
				
				}					
				execute();					
				turnRadarRightRadians(2*PI);

				state = getState();
				agent.QLearn(state, action, reward);
				accumuReward += reward;					

				reward = 0.0d;
				isHitWall = 0;
				isHitByBullet = 0;
			}
		}		
	}
	
	////=====Utility-------////////
	private void scanAndFire() {
		isFound = false;
		while(!isFound) {
			setTurnRadarLeft(360);
			execute();
		}
		
		turnGunLeft(getGunHeading() - (getHeading() + targetBearing)); //All values in degree
		double currentTargetDist = targetDist;
		if(currentTargetDist < 101) fire(6); //Super bullet
		else if(currentTargetDist < 201) fire(4);//Big Bullet
		else if(currentTargetDist < 301) fire(2);// Small Bullet
		else fire(1); //Tiny bullet
	}
	
	private int getState() {
		int heading = States.getHeading(getHeading()); //get heading in degrees
		int targetDistance = States.getTargetDistance(target.getDistance());
		int targetBearing = States.getTargetBearing(target.getBearing());
		int horizontalUnsafe = States.getHorizontalPositionUnsafe(getX(),getBattleFieldWidth());
		int verticalUnsafe = States.getVerticalPositionUnsafe(getY(), getBattleFieldHeight());
		int state = States.getStateIndex(heading, targetDistance, targetBearing, horizontalUnsafe,verticalUnsafe, isHitByBullet);
		return state;
		
	}
	
	// This function transform the range of bearing from 0-2pi to -pi-pi
	private double NormalizeBearing(double bearing) {
		while(bearing > PI) {
			bearing -= 2*PI;
		}
		while(bearing < -PI) {
			bearing += 2*PI;
		}
		return bearing;
	}
	
	//=======Robot Events=======////

	public void onScannedRobot(ScannedRobotEvent e) {
		isFound = true;
		targetDist = e.getDistance();
		targetBearing = e.getBearing();
		if ((e.getDistance() < target.getDistance())||(target.getName() == e.getName()))   
        {   
          double absbearing_rad = (getHeadingRadians()+e.getBearingRadians())%(2*PI);   
          target.setName(e.getName());   
          double h = NormalizeBearing(e.getHeadingRadians() - target.getHead()); 
          h = h/(getTime() - target.getCtime());   
          target.setChangeHead(h);   
          target.setPositionX(getX()+Math.sin(absbearing_rad)*e.getDistance());   
          target.setPositionY(getY()+Math.cos(absbearing_rad)*e.getDistance());   
          target.setBearing(e.getBearingRadians());   
          target.setHead(e.getHeadingRadians());  
          target.setCtime(getTime());             
          target.setSpeed(e.getVelocity());  
          target.setDistance(e.getDistance());   
          target.setEnergy(e.getEnergy());   
        }
	}

	
	public void onBulletHit(BulletHitEvent e)   
    {  
		if (target.getName() == e.getName()) {     
		    double change = e.getBullet().getPower() * 9;   
		    System.out.println("Bullet Hit: " + change);   
		    if (interRewards) reward += change;   
		}   
    }  
	
	

	public void onBulletMissed(BulletMissedEvent e)   
    {   
		double change = -e.getBullet().getPower() * 7.5;   
		System.out.println("Bullet Missed: " + change);   
		if (interRewards) reward += change;   
    }

	public void onHitByBullet(HitByBulletEvent e) {
		if (target.getName()== e.getName())   {   
			double power = e.getBullet().getPower();   
			double change = -6 * power;
			System.out.println("Hit By Bullet: " + change);   
			if (interRewards) reward += change;  
		}
		isHitByBullet = 1;  
	}
	

	public void onHitRobot(HitRobotEvent e) {   
		if (target.getName() == e.getName()) {   
			double change = -6.0;   
			System.out.println("Hit Robot: " + change);   
			if (interRewards) reward += change;   
		}   
    }  
	

	public void onHitWall(HitWallEvent e) {
		double change = -10.0;   
		System.out.println("Hit Wall: " + change);   
		if (interRewards) reward += change;   
        isHitWall = 1;
      
	}	
	

	public void onRobotDeath(RobotDeathEvent e) {   
		if (e.getName() == target.getName()) {
			target.setDistance(10000); 
		}
		if (interRewards) reward += 20;
    }   
	

	public void onWin(WinEvent event)   
    {   
		reward+=rewardForWin;
		if(!isOnline) saveData();   
  		int winningTag=1;
  		if(isOnline) {
	  		for(NeuralNet net : agent.getNeuralNetworks())
	  		{
	  			net.save_robot(getDataFile("Weight_"+net.getNetID()+".dat"));
	  		}
  		}
  		PrintStream w = null; 
  		try { 
  			w = new PrintStream(new RobocodeFileOutputStream(getDataFile("battle_history.dat").getAbsolutePath(), true)); 
  			w.println(getRoundNum()+" \t"+winningTag);
  			if (w.checkError()) 
  				System.out.println("Could not save the data!");  
  				w.close(); 
  		} 
	    catch (IOException e) { 
	    	System.out.println("IOException trying to write: " + e); 
	    } 
	    finally { 
	    	try { 
	    		if (w != null) 
	    			w.close(); 
	    	} 
	    	catch (Exception e) { 
	    		System.out.println("Exception trying to close writer: " + e); 
	    	}
	    } 
    }   
     

    public void onDeath(DeathEvent event)   
    {   
    	reward+=rewardForDeath;
    	if(!isOnline) saveData();  
       
		int losingTag=0;
  		if(isOnline) {
	  		for(NeuralNet net : agent.getNeuralNetworks())
	  		{
	  			net.save_robot(getDataFile("Weight_"+net.getNetID()+".dat"));
	  		}
  		}
		PrintStream w = null; 
		try { 
			w = new PrintStream(new RobocodeFileOutputStream(getDataFile("battle_history.dat").getAbsolutePath(), true)); 
			w.println(getRoundNum()+" \t"+losingTag);
			if (w.checkError()) 
				System.out.println("Could not save the data!"); 
			w.close(); 
		} 
		catch (IOException e) { 
			System.out.println("IOException trying to write: " + e); 
		} 
		finally { 
			try { 
				if (w != null) 
					w.close(); 
			} 
			catch (Exception e) { 
				System.out.println("Exception trying to close writer: " + e); 
			} 
		} 
    }	
    

	public void loadData()   {   
	    try   {   
	      lut.loadData(getDataFile("LUT.dat"));   
	    }   
	    catch (Exception e)   {
	    	out.println("Exception trying to write: " + e); 
	    }   
	}   
	     
	public void saveData()   {   
	    try   {   
	      lut.saveData(getDataFile("LUT.dat"));
	    }   
	    catch (Exception e)   {   
	      out.println("Exception trying to write: " + e);   
	    }   
	 }
	
	public void printQValueError(){
  		PrintStream save = null; 
  		try { 
  			save = new PrintStream(new RobocodeFileOutputStream(getDataFile("QValueVariation.dat").getAbsolutePath(), true)); //Print at the end of file
  			save.println(agent.getErrorSignal());
  			if (save.checkError()) 
  				System.out.println("Save error signal failed!"); 
  				save.close(); 
  		} 
	    catch (IOException e) { 
	    	System.out.println("Error when trying to write: " + e); 
	    } 
	    finally { 
	    	try { 
	    		if (save != null) 
	    			save.close(); 
	    	} 
	    	catch (Exception e) { 
	    		System.out.println("Finally error when trying to close writer: " + e); 
	    	}
	    } 
	}
}
