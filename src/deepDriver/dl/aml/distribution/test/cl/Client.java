package deepDriver.dl.aml.distribution.test.cl;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;

public class Client {
	public void run() throws  Exception {
		System.out.println("employeeNumber= "
                + joe .getEmployeeNumber());
System.out.println("employeeName= "
                + joe .getEmployeeName());

Socket socketConnection = new Socket("127.0.0.1", 11111);


ObjectOutputStream clientOutputStream = new
ObjectOutputStream(socketConnection.getOutputStream());
ObjectInputStream clientInputStream = new 
ObjectInputStream(socketConnection.getInputStream());

clientOutputStream.writeObject(joe);
clientOutputStream.flush();

Employee joe2 = (Employee)clientInputStream.readObject();
System.out.println(joe);
System.out.println(joe2);
System.out.println("employeeNumber= "
                + joe2 .getEmployeeNumber());
System.out.println("employeeName= "
                + joe2 .getEmployeeName());
joe2.setEmployeeName("aaaa");
joe2.setEmployeeNumber(1);
clientOutputStream.writeObject(joe2);
clientOutputStream.flush();

joe2 = (Employee)clientInputStream.readObject();
System.out.println(joe);
System.out.println(joe2);
System.out.println("employeeNumber= "
                + joe2 .getEmployeeNumber());
System.out.println("employeeName= "
                + joe2 .getEmployeeName());
joe2.setEmployeeName("bbbb");
joe2.setEmployeeNumber(2);
clientOutputStream.writeObject(joe2);
clientOutputStream.flush();

clientOutputStream.close();
clientInputStream.close();
	}
	Employee joe = new Employee(150, "Joe");

	   public static void main(String[] arg) {
	      try {
	         
	    	  Client client = new Client(); 
	    	  client.run();
	         

	      } catch (Exception e) {System.out.println(e); }
	   }
	}
