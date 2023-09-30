import { SupportRequestService } from './../../../../../services/support_request.service';
import { DatePipe } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { MessageService } from 'primeng/api';
import { StatusFilter } from 'src/constants/global-constants';
import { IStaffMemberResponse, StaffMember } from 'src/models/staff-member';
import {
  IAsset,
  IAssetClassesResponse,
  IAssetListResponse,
  IAssetPartnerListResponse,
  IName,
} from 'src/models/asset';
import { AssetService } from 'src/services/asset.service';
@Component({
  selector: 'app-asset-list',
  templateUrl: './asset-list.component.html',
  styleUrls: ['./asset-list.component.scss'],
})
export class AssetListComponent implements OnInit {
  loading: boolean = true;
  assetList: Array<IAsset> = [];
  sr_List: Array<StaffMember> = [];
  selectedStaffMember: StaffMember | null = null;
  statusOption: Array<string> = StatusFilter.assetListOptions;
  assetTypeOption: Array<IName> = [];
  assetPartnerOption: Array<IName> = [];
  sortOption: any[] = [
    { name: 'ASC Created_on', value: 'created_on' },
    { name: 'DESC Created_on', value: '-created_on' },
  ];
  selectedSort: any = null;
  selectedStatus: Array<string> = [];
  selectedAssetType: Array<IName> = [];
  selectedAssetPartner: Array<IName> = [];
  offset: number = 0;
  limit: number = 10;
  totalCount: number = 0;
  selectedDate: any = null;
  constructor(
    private assetService: AssetService,
    private supportRequestService: SupportRequestService,
    public messageService: MessageService,
    public datepipe: DatePipe
  ) {}
  ngOnInit() {
    this.selectedSort = this.sortOption[0];
    this.getAssetTypeList();
    this.getAssetPartnerList();
    this.getAssetList();
    this.getManagedByFilterList();
  }
  getManagedByFilterList() {
    this.supportRequestService.fetchStaffMemberList().subscribe({
      next: (res: IStaffMemberResponse) => {
        this.sr_List = res.data.staff;
        console.log('Asset Staff Memeber List Response:', res);
      },
      error: (err) => {
        console.error('Asset Staff Memeber List Error:', err);
      },
    });
  }
  getAssetTypeList() {
    this.assetService.fetchAssetClasses().subscribe({
      next: (res: IAssetClassesResponse) => {
        this.assetTypeOption = res.data;
        console.log('Asset Type List Response:', res);
      },
      error: (err) => {
        console.error('Asset Type List Error:', err);
      },
    });
  }
  getAssetPartnerList() {
    this.assetService.fetchPartnerList().subscribe({
      next: (res: IAssetPartnerListResponse) => {
        this.assetPartnerOption = res.data.partners;
        console.log('Asset Partner List Response:', res);
      },
      error: (err) => {
        console.error('Asset Partner List Error:', err);
      },
    });
  }
  getAssetList() {
    this.loading = true;
    let statuses: string = this.selectedStatus
      ? this.selectedStatus.toString()
      : '';
    let assetClasses: string = this.selectedAssetType
      ? this.selectedAssetType.map((type) => type.id).toString()
      : '';
    let partnerIds: string = this.selectedAssetPartner
      ? this.selectedAssetPartner.map((partner) => partner.id).toString()
      : '';

    let startDate;
    let endDate;
    if (
      this.selectedDate &&
      this.selectedDate[0] != null &&
      this.selectedDate[1] != null
    ) {
      startDate = this.datepipe.transform(this.selectedDate[0], 'yyyy-MM-dd');
      endDate = this.datepipe.transform(this.selectedDate[1], 'yyyy-MM-dd');
    } else {
      startDate = '';
      endDate = '';
    }
    this.assetService
      .fetchAssetList(
        this.offset,
        this.limit,
        startDate,
        endDate,
        statuses,
        assetClasses,
        partnerIds,
        this.selectedSort.value
      )
      .subscribe({
        next: (res: IAssetListResponse) => {
          this.totalCount = res.data.pagination_count;
          this.assetList = res.data.list;
          console.log('Asset List Response:', res);
        },
        error: (err) => {
          this.messageService.add({
            severity: 'error',
            summary: 'Something went wrong',
            detail: err.error.message,
          });
          console.error('Asset List Error:', err);
        },
        complete: () => {
          this.loading = false;
        },
      });
  }
  changeFilterDate() {
    if (this.selectedDate[0] != null && this.selectedDate[1] != null) {
      this.offset = 0;
      this.getAssetList();
    }
  }
  filterChange() {
    this.offset = 0;
    this.getAssetList();
  }
  clearFilter() {
    this.selectedStatus = [];
    this.selectedAssetType = [];
    this.selectedAssetPartner = [];
    this.selectedSort = this.sortOption[0];
    this.selectedDate = null;
    this.offset = 0;
    this.limit = 10;
    this.getAssetList();
  }
  paginate(event: any) {
    this.offset = event.first;
    this.limit = event.rows;
    this.getAssetList();
  }
}
