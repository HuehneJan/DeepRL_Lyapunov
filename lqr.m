
A = [1.05 0.05; 0  1.05]
B = [0; 0.05]
Q = [1 0; 0 1]
R = 1

[K,S,e] = dlqr(A,B,Q,R)
disp("Linear System 1")
disp("Open-Loop Eigenvalues:")
disp(eig(A))
disp('Optimal Feedback Matrix K:')
disp(K)
disp("Closed-Loop Eigenvalues:")
disp(e)

A = [1 0.1; -0.05  1]
B = [0.05; 0]
[K,S,e] = dlqr(A,B,Q,R)

disp("Linear System 2")
disp("Open-Loop Eigenvalues:")
disp(eig(A))
disp('Optimal Feedback Matrix K:')
disp(K)
disp("Closed-Loop Eigenvalues:")
disp(e)



A = [1 0.05;  0.7358 0]
B = [0; 0.15]

[K,S,e] = dlqr(A,B,Q,R)
disp("Linearized Pendulum System")
disp("Open-Loop Eigenvalues:")
disp(eig(A))
disp('Optimal Feedback Matrix K:')
disp(K)
disp("Closed-Loop Eigenvalues:")
disp(e)





