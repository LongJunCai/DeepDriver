package com.test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.test.students.Student;

public class TestCollection {
	
	public static void main(String[] args) {
		List<Integer> ll = new ArrayList<Integer>();
		ll.add(77);
		for (int i = 0; i < 1000; i++) {
			ll.add(i);
		}
//		for (int i = 0; i < ll.size(); i++) {
//			System.out.println(ll.get(i));
//		}
		
		Map<Integer, Student> map = new HashMap<Integer, Student>();
		List<Student> stus = new ArrayList<Student>();
		Student s1 = new Student();
		s1.setId(1);
		s1.setName("pizicai");
		stus.add(s1);
		for (int i = 0; i < stus.size(); i++) {
			Student stu = stus.get(i);
			if (stu.getId() == 30) {
				System.out.println(stu.getName());
			}
		}
		
		map.put(s1.getId(), s1);
		Student stu = map.get(30);
		System.out.println(stu.getName());
		
	}

}
