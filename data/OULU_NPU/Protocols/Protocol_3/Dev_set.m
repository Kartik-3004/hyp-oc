clear all;
clc
Files=[1,2,3,4,5]
%% Protocol 3
for phone=1:6  
      cpt=1;
      for p=1:6
          if (p~=phone)
            for s=1:3
              for u=21:35
                 for k=Files
                      filname=strcat(num2str(p),'_',num2str(s),'_',num2str(u,'%02d'),'_',num2str(k));
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
      end
  dlmwrite(strcat('Dev_',num2str(phone),'.txt'),file_names,'delimiter',''); 
  clear file_names;
end