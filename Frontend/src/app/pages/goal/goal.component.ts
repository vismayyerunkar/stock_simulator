import { Component, OnInit } from '@angular/core';
import { FormControl, FormGroup } from '@angular/forms';
import { MessageService, SelectItem } from 'primeng/api';
import { Asset } from 'src/constants/global-constants';
import { AssetService } from 'src/services/asset.service';
interface City {
  name: string;
  code: string;
}
@Component({
  selector: 'app-goal',
  templateUrl: './goal.component.html',
  styleUrls: ['./goal.component.scss'],
})
export class GoalComponent implements OnInit {
  riskPercentage: number; // Initialize the risk percentage
  riskMeterWidth: number = 0; // Width of the risk meter fill
  isDragging: boolean = false;
  checked: boolean = false;
  formGroup: FormGroup | undefined;
  Invested_Amount: number ;
  Future_Value: number ;
  Risk: number  =0;
  No_Year:Number  ;
  cities: City[] | undefined;
  selectedCity: City | undefined;
  displayGoalAssets = false;
  goalAssets: any = [];
  goalDetails: any = {};
  RateOfReturn:number ;


  constructor(private messageService: MessageService, private assestService: AssetService ) 
  {

  }

  
  startDragging(event: MouseEvent) {
    this.isDragging = true;
    this.updateRiskMeterWidth(event);
  }

  InvestedAmountChange(data : any){
    this.Invested_Amount = data
  }

  GetFutureValue(data : any){
    this.Future_Value = data
  }

  GetNumberOfYear(data :any){
    this.No_Year = data 
  }

  GetRateOfReturn(data:any){
    this.RateOfReturn = data
  }

  getNewGoal(){
    if(!this.Future_Value || !this.riskPercentage || !this.No_Year || !this.Invested_Amount || !this.RateOfReturn){
      return alert("Please fill all input fields")
    }
    console.log({pv:this.Invested_Amount,                  
      fv:this.Future_Value,                  
      r : this.riskPercentage,
      n:this.No_Year ?? 0 
      });
    (this.assestService.getNewGoalDetails({pv:this.Invested_Amount,                  
      fv:this.Future_Value,                  
      r : this.RateOfReturn,
      n:this.No_Year ?? 0 ,
      risk:this.riskPercentage
      }))?.subscribe({
      next : (data : any)=>
      {
        this.goalDetails =data[1]
        this.goalAssets = JSON.parse(data[0])
        console.log("new goal", this.goalAssets)
        console.log("Asset", this.goalDetails)
        this.show()
    }

      , error :(err: any)=> console.error('Error:', err)
      
    });
  }
  //

  stopDragging() {
    this.isDragging = false;
  }

  show() {
    this.messageService.add({
      severity: 'success',
      summary: 'Goal Added',
      detail: 'Richin',
    });
    this.displayGoalAssets = true;
  }

  updateRiskMeterWidth(event: MouseEvent) {
    if (this.isDragging) {
      const containerRect = document
        .querySelector('.risk-meter-bar')
        ?.getBoundingClientRect();
      if (containerRect) {
        const mouseX = event.clientX - containerRect.left;
        const maxWidth = containerRect.width;
        const newWidth = Math.min(maxWidth, Math.max(0, mouseX));
        this.riskMeterWidth = (newWidth / maxWidth) * 100;
        this.riskPercentage = Math.round((newWidth / maxWidth) * 100);
      }
    }
  }

  goalPeriodOptions: SelectItem[] = [
    { label: '6 Months', value: 1 },
    { label: '1 Year', value: 2 },
    { label: '2 Year', value: 3 },
    { label: '3 Year', value: 4 },
    { label: '4 Year', value: 5 },
    { label: '5 Year', value: 6 },
    { label: '6 Year', value: 7 },
    { label: '7 Year', value: 8 },
  ];

  selectedGoalPeriod: number | null = null;

  ngOnInit(): void {
    this.cities = [
      { name: 'New York', code: 'NY' },
      { name: 'Rome', code: 'RM' },
      { name: 'London', code: 'LDN' },
      { name: 'Istanbul', code: 'IST' },
      { name: 'Paris', code: 'PRS' },
    ];

    this.formGroup = new FormGroup({
      selectedCity: new FormControl<City | null>(null),
    });
  }
}
