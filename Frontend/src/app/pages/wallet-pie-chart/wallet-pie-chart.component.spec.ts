import { ComponentFixture, TestBed } from '@angular/core/testing';

import { WalletPieChartComponent } from './wallet-pie-chart.component';

describe('WalletPieChartComponent', () => {
  let component: WalletPieChartComponent;
  let fixture: ComponentFixture<WalletPieChartComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ WalletPieChartComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(WalletPieChartComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
