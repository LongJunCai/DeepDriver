package deepDriver.dl.aml.distribution.test.cl;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.net.Socket;

public class Server {
	 Employee employee = null;
	public void run() throws Exception {
		 ServerSocket socketConnection = new ServerSocket(11111);

         System.out.println("Server Waiting");

         Socket pipe = socketConnection.accept();

         ObjectOutputStream serverOutputStream = new 
            ObjectOutputStream(pipe.getOutputStream());
         
         ObjectInputStream serverInputStream = new    
            ObjectInputStream(pipe.getInputStream());

         employee = (Employee )serverInputStream.readObject();
         System.out.println("employeeNumber= "
                 + employee .getEmployeeNumber());
 System.out.println("employeeName= "
                 + employee .getEmployeeName());

         employee .setEmployeeNumber(256);
         employee .setEmployeeName("John");

         serverOutputStream.writeObject(employee);
         serverOutputStream.flush();
         
         employee = (Employee )serverInputStream.readObject();
         System.out.println("employeeNumber= "
                 + employee .getEmployeeNumber());
 System.out.println("employeeName= "
                 + employee .getEmployeeName());
 	serverOutputStream.writeObject(employee);
 	serverOutputStream.flush();
 	employee = (Employee )serverInputStream.readObject();
 System.out.println("employeeNumber= "
         + employee .getEmployeeNumber());
System.out.println("employeeName= "
         + employee .getEmployeeName());
         
         serverInputStream.close();
         serverOutputStream.close();
	}

	   public static void main(String[] arg) {

	     

	      try {
	    	  Server server = new Server();
	    	  server.run();      


	      }  catch(Exception e) {System.out.println(e); 
	      }
	   }

	}