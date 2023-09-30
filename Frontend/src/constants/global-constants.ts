export class StatusFilter {
  public static options: Array<string> = [
    'COMMITTED',
    'PROCESSING',
    'DISBURSED',
    'CLOSED',
    'CANCELED',
  ];
  public static supportRequestStatus: Array<string> = [
    'NEW',
    'IN-PROGRESS',
    'PROCESSING',
    'CANCELLED',
    'COMPLETED',
    'CLOSED',
  ];
  public static changeSupportRequestOption: Array<string> = [
    'NEW',
    'IN-PROGRESS',
    'CANCELLED',
    'CLOSED',
  ];
  public static assetListOptions: Array<string> = [
    'NEW',
    'LISTED',
    'PRE-LISTED',
    'SUBSCRIBED',
    'CANCELLED',
    'DISBURSED',
  ];
  public static withdrawListOptions: Array<string> = ['CREDITED ', 'DEBITED'];
}

export class Report {
  public static types: Array<string> = [
    'INVESTMENT_REPORT',
    'INCOME_REPORT',
    'ACCOUNT_STATEMENT',
  ];
}

export class Asset {
  public static tenureTypes: Array<string> = ['DAILY', 'MONTHLY', 'YEARLY'];
  public static repaymentTypes: Array<string> = [
    'MONTHLY',
    'BIMONTHLY',
    'QUARTERLY',
    'HALF-YEARLY',
    'YEARLY',
    'ON-MATURITY',
  ];
  public static riskCategoryList: Array<string> = ['LOW', 'MEDIUM', 'HIGH'];
  public static acceptedInvestmentTypes: Array<string> = [
    'SMART INVEST',
    'FIXED DEPOSIT',
    'SIP',
  ];
}
