import { DatePipe, Location } from '@angular/common';
import { Component, OnInit, Pipe, PipeTransform } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { MessageService } from 'primeng/api';
import { Asset } from 'src/constants/global-constants';
import {
  IAddOrUpdateAssetResponse,
  IAssetClassesResponse,
  IAssetDetail,
  IAssetDetailResponse,
  IAssetPartnerListResponse,
  IName,
  Property,
} from 'src/models/asset';
import { AssetService } from 'src/services/asset.service';
import { UserAuthService } from 'src/services/user-auth.service';

@Component({
  selector: 'app-add-asset',
  templateUrl: './add-asset.component.html',
  styleUrls: ['./add-asset.component.scss'],
})
export class AddAssetComponent implements OnInit {
  assetId: string = '';
  asset: IAssetDetail;

  fileAsBlog: Blob | undefined;
  file: any = null;
  fileAsBase64: any = null;
  loading: boolean = false;

  assetPartnerOption: Array<IName> = [];
  assetClassOptions: Array<IName> = [];
  tenureOptions: Array<string> = Asset.tenureTypes;
  repaymentOptions: Array<string> = Asset.repaymentTypes;
  riskCategoryOptions: Array<string> = Asset.riskCategoryList;
  acceptedInvestmentOptions: Array<string> = Asset.acceptedInvestmentTypes;
  today = new Date();

  assetForm = new FormGroup({
    partner: new FormControl<number | null>(null, [Validators.required]),
    asset_class: new FormControl<number | null>(null, [Validators.required]),
    name: new FormControl('', [Validators.required]),
    short_description: new FormControl('', [Validators.required, Validators.minLength(1), Validators.maxLength(50)]),
    description: new FormControl('', [Validators.required]),
    interest_rate: new FormControl<number | null>(null, [
      Validators.required,
      Validators.min(1),
    ]),
    tenure: new FormControl<number | null>(null, [
      Validators.required,
      Validators.min(1),
    ]),
    tenure_type: new FormControl('', [Validators.required]),
    repayment_type: new FormControl('', [Validators.required]),
    minimum_investment: new FormControl<number | null>(null, [
      Validators.required,
      Validators.min(1),
    ]),
    is_secured: new FormControl(false),
    is_listed: new FormControl(false),
    maturity_date: new FormControl(this.assetId ? new Date() : '', [
      Validators.required,
    ]),
    maturity_date_buffer: new FormControl<number | null>(null, [
      Validators.required,
      Validators.min(1),
    ]),
    amount: new FormControl<number | null>(null, [
      Validators.required,
      Validators.min(1),
    ]),
    risk_category: new FormControl('', [Validators.required]),
    accepted_investment_type: new FormControl<Array<string>>(
      [],
      [Validators.required]
    ),
  });
  additionalProperties: Array<Property> = [];

  constructor(
    private location: Location,
    private _Activatedroute: ActivatedRoute,
    public datepipe: DatePipe,
    private userAuthService: UserAuthService,
    public messageService: MessageService,
    private assetService: AssetService
  ) {}

  ngOnInit(): void {
    this._Activatedroute.queryParams.subscribe((params) => {
      this.assetId = params['assetId'];
      this.getAssetPartnerList();
      this.getAssetClassesList();
      if (this.assetId) {
        this.getAssetDetails();
      }
    });
  }

  navigateBack() {
    this.location.back();
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

  getAssetClassesList() {
    this.assetService.fetchAssetClasses().subscribe({
      next: (res: IAssetClassesResponse) => {
        this.assetClassOptions = res.data;
        console.log('Asset Classes List Response:', res);
      },
      error: (err) => {
        console.error('Asset Classes List Error:', err);
      },
    });
  }

  fileUploaded(event: any) {
    this.file = event.target.files[0];
    this.file.arrayBuffer().then((arrayBuffer: any) => {
      this.fileAsBlog = new Blob([new Uint8Array(arrayBuffer)], {
        type: this.file.type,
      });
      const reader = new FileReader();
      reader.onloadend = () => {
        this.fileAsBase64 = reader.result;
      };
      reader.readAsDataURL(this.fileAsBlog);
    });
  }

  resetFile() {
    this.file = null;
    this.fileAsBlog = undefined;
    this.fileAsBase64 = null;
  }

  submitForm(type: string) {
    this.loading = true;
    let data = this.assetForm.value;

    if (data.minimum_investment && data.minimum_investment % 500 != 0) {
      this.messageService.add({
        severity: 'error',
        summary: 'Something went wrong',
        detail: 'Minimum investment must be in multiples of 500 !',
      });
      this.loading = false;
      return;
    }

    let formData = new FormData();
    if (this.assetId) {
      formData.append('id', String(this.assetId));
    }
    formData.append('partner_id', String(data.partner));
    formData.append('asset_class_id', String(data.asset_class));
    formData.append('name', String(data.name));
    formData.append('short_description', String(data.short_description));
    formData.append('description', String(data.description));
    formData.append('interest_rate', String(data.interest_rate));
    formData.append('tenure', String(data.tenure));
    formData.append('tenure_type', String(data.tenure_type));
    formData.append('repayment_type', String(data.repayment_type));
    formData.append('minimum_investment', String(data.minimum_investment));
    formData.append('is_secured', String(data.is_secured));
    formData.append('is_listed', String(data.is_listed));
    formData.append(
      'maturity_date',
      this.datepipe.transform(data.maturity_date, 'yyyy-MM-dd') ?? ''
    );
    formData.append('maturity_date_buffer', String(data.maturity_date_buffer));
    formData.append('amount', String(data.amount));
    formData.append('risk_category', String(data.risk_category));
    formData.append(
      'accepted_investment_type',
      String(data.accepted_investment_type)
    );
    formData.append('status', type);
    formData.append('managed_by_id', '1');
    var add_Details = this.additionalProperties.reduce(
      (obj, item) => Object.assign(obj, { [item.key]: item.value }),
      {}
    );
    formData.append('additional_properties', JSON.stringify(add_Details));
    if (this.fileAsBlog) {
      formData.append('img', this.fileAsBlog, this.file.name);
    }

    this.assetService.createOrUpdateAsset(formData).subscribe({
      next: (res: IAddOrUpdateAssetResponse) => {
        this.messageService.add({
          severity: 'success',
          summary: 'Congratulations',
          detail: res.message,
        });
        console.log(res.message);
      },
      error: (err) => {
        this.messageService.add({
          severity: 'error',
          summary: 'Something went wrong',
          detail: err.error.message,
        });
        this.loading = false;
        console.error('Create/Update asset error:', err);
      },
      complete: () => {
        if (!this.assetId) {
          this.resetForm();
        }
        this.loading = false;
      },
    });
  }

  getAssetDetails() {
    this.assetService.fetchAssetDetails(this.assetId).subscribe({
      next: (res: IAssetDetailResponse) => {
        this.populateForm(res.data);
        console.log('Asset Detail:', res);
      },
      error: (err) => {
        console.error('Asset Detail Error:', err);
      },
    });
  }

  populateForm(data: IAssetDetail) {
    this.assetForm.controls.partner.setValue(data.partner.id);
    this.assetForm.controls.name.setValue(data.name);
    this.assetForm.controls.short_description.setValue(data.short_description);
    this.assetForm.controls.description.setValue(data.description);
    this.assetForm.controls.interest_rate.setValue(Number(data.interest_rate));
    this.assetForm.controls.tenure.setValue(data.tenure);
    this.assetForm.controls.tenure_type.setValue(data.tenure_type);
    this.assetForm.controls.repayment_type.setValue(data.repayment_frequency);
    this.assetForm.controls.minimum_investment.setValue(
      Number(data.minimum_investment)
    );
    this.assetForm.controls.is_secured.setValue(data.is_secured);
    this.assetForm.controls.is_listed.setValue(data.is_listed);
    this.assetForm.controls.maturity_date.setValue(
      new Date(data.maturity_date)
    );
    this.assetForm.controls.maturity_date_buffer.setValue(
      data.maturity_date_buffer
    );
    this.assetForm.controls.risk_category.setValue(data.risk_category);
    this.assetForm.controls.accepted_investment_type.setValue(
      data.accepted_investment_types
    );
    this.assetForm.controls.amount.setValue(Number(data.amount));
    this.fileAsBase64 = data.image;
    for (const [key, value] of Object.entries(data.additional_properties)) {
      this.additionalProperties.push({ key: key, value: value });
    }
    this.assetClassOptions.forEach((option) => {
      if (option.name === data.asset_class.name) {
        this.assetForm.controls.asset_class.setValue(option.id);
      }
    });
  }

  resetForm() {
    this.assetForm.reset();
    this.resetFile();
    this.additionalProperties = [];
  }

  addPropertyRow() {
    this.additionalProperties.push({ key: '', value: '' });
  }

  removeProperty(index: number) {
    this.additionalProperties = this.additionalProperties.filter(
      (item, i) => i != index
    );
  }

  validateAdditionalProperty(): boolean {
    var valid: boolean = true;
    this.additionalProperties.forEach((item) => {
      if (item.key == '' || item.value == '') {
        valid = false;
      }
    });
    return valid;
  }

  omitSpecialCharacter(event: any) {
    var k = event.charCode;
    return (
      (k > 64 && k < 91) ||
      (k > 96 && k < 123) ||
      k == 32 ||
      k == 95 ||
      k == 45 ||
      (k >= 48 && k <= 57)
    );
  }
}
