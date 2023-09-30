import { AssetService } from 'src/services/asset.service';
import { Component, OnChanges, OnInit,Input, SimpleChanges } from '@angular/core';
import { MessageService } from 'primeng/api';

@Component({
  selector: 'app-goal-assests',
  templateUrl: './goal-assests.component.html',
  styleUrls: ['./goal-assests.component.scss'],
})

export class GoalAssestsComponent implements OnInit, OnChanges{
  @Input() goalAssets: any;
  @Input() goalDetails: any;
  constructor(private messageService: MessageService,private assestService: AssetService) {}

  ngOnInit(): void {}
  ngOnChanges(changes: SimpleChanges){
    console.log("current value",this.goalAssets);
  }

  createGoal(){
    (this.assestService.createGoal({
      goalAssets:this.goalAssets,
      goalDetails:this.goalDetails
      }))?.subscribe({
      next : (data : any)=>
      {
       console.log(data);
    }
      , error :(err: any)=> console.error('Error:', err)
    });
  }

  show() {
    this.messageService.add({
      severity: 'success',
      summary: 'Goal Added',
      detail: 'Richin',
    });
  }
}
