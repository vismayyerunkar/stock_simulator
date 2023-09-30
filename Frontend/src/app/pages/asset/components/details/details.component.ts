import { Component, Input, OnInit } from '@angular/core';
import { IAssetDetail } from 'src/models/asset';

@Component({
  selector: 'app-details',
  templateUrl: './details.component.html',
  styleUrls: ['./details.component.scss']
})
export class DetailsComponent implements OnInit {

  @Input() asset: IAssetDetail;

  constructor() { }

  ngOnInit(): void {
  }

}
