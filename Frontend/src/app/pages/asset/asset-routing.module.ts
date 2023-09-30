import { NotesComponent } from './components/notes/notes.component';
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AssetListComponent } from './components/asset-list/asset-list.component';
import { AssetDetailComponent } from './components/asset-detail/asset-detail.component';
import { AddAssetComponent } from './components/add-asset/add-asset.component';

const routes: Routes = [
  { path: '', redirectTo: 'list', pathMatch: 'full' },
  { path: 'list', component: AssetListComponent, pathMatch: 'full' },
  { path: 'list/:id', component: AssetDetailComponent, pathMatch: 'full' },
  { path: 'add', component: AddAssetComponent, pathMatch: 'full' },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class AssetRoutingModule {}
