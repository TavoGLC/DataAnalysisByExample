//Analisis completo de los datos de calcio desde importar los datos hasta el ajuste 
//a la exponencial simple, muestra graficas de cada uno de los pasos mas importantes 
//durante el analisis de los datos. 
//Condiciones Necesarias: para que el programa funciones es necesario realizar un poco 
//de edicion al archivo original 
//1 Abrir con clampfit el registro y guardarlo como ATF (AxonTextFile)
//2 Eliminar la extencion y cambiarla a csv 
//3 Abrir con excel y eliminar la primera parte del archivo aprox 10 lineas que contienen 
//informacion tecnica del archivo 
//4 Guardar los cambios y esta listo para ser analisado con la siguiente rutina
//****************************Notas********************************
// primera parte del analisis se separa el calculo de las metricas correspondientes 
// solo se calcula la fluorescencia relativa y el calcio libre intracelular 
// se exportan a un nuevo archivo 

function [Fig0,Fig1]=PREP(Txt)

Name=Txt;
SeparationPoint=strindex(Name,'\');
FileName=part(Name,SeparationPoint(length(SeparationPoint))+1:length(Name));

WorkingData=csvRead(Name,';');
TimeData=WorkingData(:,1);
FData=WorkingData(:,2);
TimeStep=TimeData(2)-TimeData(1);

//Filtrando la señal 

CutF=25;
FFilt=2*%pi*CutF*TimeStep;

TransferFunction=iir(2,'lp','butt',FFilt,[0.1,0.01]);
Num=TransferFunction(2);
Den=TransferFunction(3);
Fluo3Data=filter(Num,Den,FData)

clear WorkingData Name SeparationPoint FData TransferFunction Num Den CutF FFilt

//Ajustando linea base 

c=zeros(2000,1);
d=zeros(2000,1);
MeanFactor=100;

for j=1:length(c)
    a=j*MeanFactor;
    c(j)=mean(Fluo3Data(1:a));
    d(j)=mean(Fluo3Data(length(Fluo3Data)-a:length(Fluo3Data)));
end

clear a 

MeansBaseA=zeros(length(c),1);
MeansBaseB=zeros(length(d),1);

for j=1:length(c)-1
    MeansBaseA(j)=(c(j+1)-c(j))/c(j);
    MeansBaseB(j)=(d(j+1)-d(j))/d(j);
end

clear c

DiffDataA=zeros(length(MeansBaseA),1);
DiffDataB=zeros(length(MeansBaseB),1);

for j=1:length(MeansBaseA)-1
    DiffDataA(j)=MeansBaseA(j)-MeansBaseA(j+1);
    DiffDataB(j)=MeansBaseB(j)-MeansBaseB(j+1);
end

// separando la grafica 

CuttingPointA=min(DiffDataA);
CuttingPointB=min(DiffDataB);
k=1;
w=1;

while DiffDataA(k)~=CuttingPointA
    k=k+1
end

while DiffDataB(w)~=CuttingPointB
    w=w+1
end

CuttingPositionA=k*MeanFactor
CuttingPositionB=length(Fluo3Data)-(w*MeanFactor);

//Ajustando la linea Base y eliminacion de la apertura del paso de luz 

BaseLine=min(Fluo3Data(1:floor((CuttingPositionA)*0.75)));
CutFluo3DataAB=Fluo3Data(CuttingPositionA:CuttingPositionB)+abs(BaseLine);
CutFluoTime=TimeStep*(1:1:length(CutFluo3DataAB))';

Fig0=scf(0)
subplot(211)
xtitle(FileName,'Tiempo (s)','Fluorescencia')
plot(TimeData,Fluo3Data,'k')

subplot(212)
xtitle(FileName,'Tiempo (s)','Fluorescencia')
plot(CutFluoTime,CutFluo3DataAB,'k')

clear MeansBaseA MeansBaseB DiffDataA DiffDataB w k d j 
clear CuttingPositionA CuttingPositionB CuttingPointA CuttingPointB BaseLine

//Seprando los picos 

clear TimeData Fluo3Data CutFluoTime MeanFactor

MaxDataValue=max(CutFluo3DataAB);
MaxValuePositions=find(CutFluo3DataAB>0.95*MaxDataValue);
i=1;
k=1;

for j=1:length(MaxValuePositions)-1
    if MaxValuePositions(j+1)-MaxValuePositions(j)<500 then
        PeakData(i,k)=MaxValuePositions(j);
        i=i+1;
    else 
        k=k+1;
        i=1;
    end
end

TotalPeaks=k;
PeakPositions=zeros(TotalPeaks,1);

clear i j k MaxDataValue MaxValuePositions

for j=1:TotalPeaks 
    a=find(PeakData(:,j)~=0);
    b=PeakData(a,j);
    c=CutFluo3DataAB(b(1):b(length(b)));
    d=max(c);
    e=find(c==d);
    PeakPositions(j)=b(1)+e(1);
end

clear a b c d e j PeakData 

//Separacion de los picos con ventana de tiempo fija 

TakenDataA=2000;
TakenDataB=10000;
FluoData=zeros(TakenDataA+TakenDataB,TotalPeaks);

for j=1:TotalPeaks
    if j==TotalPeaks then
        for k=(PeakPositions(j)-TakenDataA+1):length(CutFluo3DataAB)
            FixFactor=abs(PeakPositions(j)-TakenDataA);
            FluoData(k-FixFactor,j)=CutFluo3DataAB(k);
        end 
    else 
        for k=(PeakPositions(j)-TakenDataA+1):(PeakPositions(j)+TakenDataB)
                FixFactor=abs(PeakPositions(j)-TakenDataA);
                FluoData(k-FixFactor,j)=CutFluo3DataAB(k);
        end
    end
end

DataLength=TakenDataA+TakenDataB;

clear PeakPositions j k FixFactor TakenDataA TakenDataB

LengthData=zeros(TotalPeaks,1);

for j=1:TotalPeaks 
    LengthData(j)=length(find(FluoData(:,j)~=0));
end

GoodPeaks=find(LengthData==DataLength);

clear LengthData j CutFluo3DataAB

MeanFluoData=zeros(DataLength,1);

for j=1:DataLength
    MeanFluoData(j)=mean(FluoData(j,GoodPeaks));
end

clear GoodPeaks j
