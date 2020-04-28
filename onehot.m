function [ output ] = onehot( input )
[a,~]=size(input);
b=max(max(input))+1;
output=zeros(a,b);
for i=1:a;
    output(i,input(i)+1)=1;
end
end

