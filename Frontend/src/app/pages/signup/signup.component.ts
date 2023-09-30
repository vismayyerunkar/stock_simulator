import { Component, OnInit } from '@angular/core';
import {
  FormBuilder,
  FormControl,
  FormGroup,
  Validators,
} from '@angular/forms';
import { Router } from '@angular/router';
import { MessageService } from 'primeng/api';
import { ISignupResponse } from 'src/models/SignUpModel';
import { UserAuthService } from 'src/services/user-auth.service';

@Component({
  selector: 'app-signup',
  templateUrl: './signup.component.html',
  styleUrls: ['./signup.component.scss'],
})
export class SignupComponent implements OnInit {
  signupForm = new FormGroup({
    email: new FormControl('', [Validators.required, Validators.email]),
    name: new FormControl('', [Validators.required, Validators.minLength(6)]),
    phone: new FormControl('', [Validators.required, Validators.minLength(6)]),
    password: new FormControl('', [
      Validators.required,
      Validators.minLength(6),
    ]),
    rpassword: new FormControl('', [
      Validators.required,
      Validators.minLength(6),
    ]),
  });

  constructor(
    private formBuilder: FormBuilder,
    private router: Router,
    private userAuthService: UserAuthService,
    public messageService: MessageService
  ) {}

  ngOnInit() {}
  signup() {
    var body = this.signupForm.value;
    console.log(body)
    this.userAuthService.signup(body).subscribe({
      next: (res: ISignupResponse) => {
        console.log('Login Response:', res);

        this.signupForm.reset();
        this.userAuthService.accessToken = (res as any).token;
        this.router.navigate(['/login']);
      },
      error: (err: any) => {
        this.messageService.add({
          severity: 'error',
          summary: 'Failed',
          detail: err.error.message,
        });
        console.error('Signup Error:', err);
      },
    });
  }
}
