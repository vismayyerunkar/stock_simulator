import { Component, OnInit } from '@angular/core';
import { Location } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { AssetService } from 'src/services/asset.service';
import { IAssetDetail, IAssetDetailResponse } from 'src/models/asset';
import { ConfirmationService, MessageService } from 'primeng/api';

@Component({
  selector: 'app-asset-detail',
  templateUrl: './asset-detail.component.html',
  styleUrls: ['./asset-detail.component.scss'],
  providers: [ConfirmationService, MessageService],
})
export class AssetDetailComponent implements OnInit {
  id: string;
  asset: IAssetDetail;
  loading: boolean = false;

  constructor(
    private location: Location,
    private route: ActivatedRoute,
    private assetService: AssetService,
    public messageService: MessageService,
    private confirmationService: ConfirmationService
  ) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe((params) => {
      this.id = params.get('id') ?? '';
      if (this.id) {
        this.getAssetDetails(this.id);
      }
    });
  }

  getAssetDetails(assetId: string) {
    this.loading = true;

    this.assetService.fetchAssetDetails(assetId).subscribe({
      next: (res: IAssetDetailResponse) => {
        this.asset = res.data;
        console.log('Asset Detail Response:', res);
      },
      error: (err) => {
        console.error('Asset Detail Error:', err);
      },
      complete: () => {
        this.loading = false;
      },
    });
  }

  navigateBack() {
    this.location.back();
  }

  approveAssetConfirmation() {
    this.confirmationService.confirm({
      message: 'Are you sure that you want to Approve ?',
      header: 'Approval Confirmation',
      icon: 'pi pi-exclamation-triangle',
      accept: () => this.approveAsset(),
    });
  }

  CancelAssetConfirmation() {
    this.confirmationService.confirm({
      message: 'Are you sure that you want to Cancel ?',
      header: 'Cancel Approval',
      icon: 'pi pi-exclamation-triangle',
      accept: () => this.cancelAsset(),
    });
  }
  approveAsset() {
    this.assetService.approveAsset(this.id).subscribe({
      next: (res: any) => {
        console.log(res.message);
        this.asset.status = 'LISTED';
        this.messageService.add({
          severity: 'success',
          summary: 'Congratulation',
          detail: res.message,
        });
      },
      error: (err: any) => {
        console.error(err.error.message);
        this.messageService.add({
          severity: 'error',
          summary: 'Failed',
          detail: err.error.message,
        });
      },
    });
  }
  cancelAsset() {
    this.assetService.cancelAsset(this.id).subscribe({
      next: (res: any) => {
        console.log(res.message);
        this.messageService.add({
          severity: 'success',
          summary: 'Congratulation',
          detail: res.message,
        });
      },
      error: (err: any) => {
        console.error(err.error.message);
        this.messageService.add({
          severity: 'error',
          summary: 'Failed',
          detail: err.error.message,
        });
      },
    });
  }
}
