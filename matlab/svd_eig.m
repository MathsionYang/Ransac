A = ...
[-445.17242, -455.82898, -1, 0, 0, 0, 216741.05, 221929.39, 486.86987;
 0, 0, 0, -445.17242, -455.82898, -1, 214323.03, 219453.52, 481.43826;
 646.73529, -404.85107, -1, 0, 0, 0, -209306.72, 131024.31, 323.63583;
 0, 0, 0, 646.73529, -404.85107, -1, -351841.84, 220250.16, 544.02759;
 602.4458, -395.31635, -1, 0, 0, 0, -195937.41, 128571.34, 325.23657;
 0, 0, 0, 602.4458, -395.31635, -1, -324243.28, 212763.83, 538.21155;
 -1138.7687, -573.99194, -1, 0, 0, 0, 737351.25, 371659.06, 647.49872;
 0, 0, 0, -1138.7687, -573.99194, -1, 541678.5, 273030.94, 475.67035]

[u,s,v] = svd (A);
[V,D] = eig (A'*A);

[v(:,end) V(:,1)]