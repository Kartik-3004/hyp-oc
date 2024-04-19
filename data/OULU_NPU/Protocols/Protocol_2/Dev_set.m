clear all;
clc
Files=[1,2,4]
%% Protocol_1
cpt=1
for p=1:6
    for s=1:3
        for u=21:35
             for k=Files
                  filname=strcat(num2str(p),'_',num2str(s),'_',num2str(u,'%02d'),'_',num2str(k))
                  if (k==1) 
                     file_names(cpt,:)=['+1,', filname]                   
                  else
                     file_names(cpt,:)=['-1,', filname]
                  end                   
                  cpt=cpt+1;                  
            end 
        end
     end
end
dlmwrite('Dev.txt',file_names,'delimiter','');  