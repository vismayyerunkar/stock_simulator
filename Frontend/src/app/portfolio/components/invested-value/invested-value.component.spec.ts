import { ComponentFixture, TestBed } from '@angular/core/testing';

import { InvestedValueComponent } from './invested-value.component';

describe('InvestedValueComponent', () => {
  let component: InvestedValueComponent;
  let fixture: ComponentFixture<InvestedValueComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ InvestedValueComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(InvestedValueComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
