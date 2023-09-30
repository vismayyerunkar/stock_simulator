import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GoalReportComponent } from './goal-report.component';

describe('GoalReportComponent', () => {
  let component: GoalReportComponent;
  let fixture: ComponentFixture<GoalReportComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ GoalReportComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GoalReportComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
