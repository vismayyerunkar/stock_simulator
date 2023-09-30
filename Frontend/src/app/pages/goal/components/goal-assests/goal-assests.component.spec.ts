import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GoalAssestsComponent } from './goal-assests.component';

describe('GoalAssestsComponent', () => {
  let component: GoalAssestsComponent;
  let fixture: ComponentFixture<GoalAssestsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ GoalAssestsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GoalAssestsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
