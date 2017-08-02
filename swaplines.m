function [ mat ] = swaplines( mat,i,j )

temp = mat(j,:);
mat(j,:) = mat(i,:);
mat(i,:)= temp;

end

