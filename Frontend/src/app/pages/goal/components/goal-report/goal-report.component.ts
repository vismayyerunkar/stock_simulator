import { ActivatedRoute } from '@angular/router';
import { AssetService } from 'src/services/asset.service';
import { Component, OnChanges, OnInit } from '@angular/core';
import { MessageService } from 'primeng/api';

@Component({
  selector: 'app-goal-report',
  templateUrl: './goal-report.component.html',
  styleUrls: ['./goal-report.component.scss'],
})
export class GoalReportComponent implements OnInit {
  
  goalDetails:any = {};
  goalAssets:any[]= []
  goalId:string;
  constructor(private route: ActivatedRoute,private messageService: MessageService,private assetService:AssetService) {

    this.route.params.subscribe((params: any) => {
      this.goalId = params.id;
      console.log(params);
    });

    if(this.goalId){
      this.assetService.getGoalDetails(this.goalId).subscribe((data)=>{
          this.goalAssets = data?.goalAssets
          this.goalDetails = data?.goalDetails
          console.log(data);
      })
    }
    
  }

  show() {
    this.messageService.add({
      severity: 'success',
      summary: 'Goal Created Succesfully',
      detail: '43,546.00',
    });
  }

  ngOnInit(): void {}
}
